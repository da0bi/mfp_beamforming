import json


import psysmon.core.packageNodes
import psysmon.core.result as result
import psysmon.core.preferences_manager as psy_pm

# Import GUI related modules only if wxPython is available.
if psysmon.wx_available:
    import psysmon.gui.dialog.pref_listbook as psy_lb

from obspy.core.utcdatetime import UTCDateTime


class MfpBeamform(psysmon.core.packageNodes.LooperCollectionChildNode):
    '''
    '''
    name = 'MFP beamforming'
    mode = 'looper child'
    category = 'beamforming'
    tags = ['array', 'beamforming']

    def __init__(self, **args):
        psysmon.core.packageNodes.LooperCollectionChildNode.__init__(self, **args)

        self.create_parameters_prefs()


        # The processing parameters loaded from a json file.
        self.process_params = None
        
        # The computed beam data.
        self.beam_data = {}

        # Last day of the saved beam data.
        self.save_day = {}

        # The interval for which to create the results. This is ignored when
        # processing events. For events a result is created for each event.
        # TODO: Make the save interval user selectable.
        self.save_interval = 3600.


    def create_parameters_prefs(self):
        ''' Create the preference items of the parameters section.
        '''
        par_page = self.pref_manager.add_page('parameters')
        file_group = par_page.add_group('input file')

        item = psy_pm.FileBrowsePrefItem(name = 'params_filename',
                                         label = 'params file',
                                         value = '',
                                         tool_tip = 'The json file containing the processing parameters.')
        file_group.add_item(item)


    def edit(self):
        ''' Show the node edit dialog.
        '''
        dlg = psy_lb.ListbookPrefDialog(preferences = self.pref_manager)
        dlg.ShowModal()
        dlg.Destroy()
    
    #########################################################################      
    # START of matched-field beamforming functions
    # from https://github.com/fablindner/glseis/blob/master/array_analysis.py
    #########################################################################
    def calculate_CSDM(dft_array, neig=0, norm=True):
        """
        Calculate CSDM matrix for beamforming.
        :param dft_array: 2-Dim array containing DFTs of all stations
            and for multiple time windows. dim: [number stations, number windows]
        :param neig: Number of eigenvalues to project out.
        :param norm: If True, normalize CSDM matrix.
        """
        # CSDM matrix
        K = np.dot(dft_array, dft_array.conj().T)
        #if np.linalg.matrix_rank(K) < dft_array.shape[0]:
        #    warnings.warn("Warning! Poorly conditioned cross-spectral-density matrix.")

        # annul dominant source
        if neig > 0:
            K = annul_dominant_interferers(K, neig, dft_array)

        # normalize
        if norm:
            K /= np.linalg.norm(K)

        return K
    
    def annul_dominant_interferers(CSDM, neig, data):
        """
        This routine cancels the strong interferers from the data by projecting the
        dominant eigenvectors of the cross-spectral-density matrix out of the data.
        :type CSDM: numpy.ndarray
        :param CSDM: cross-spectral-density matrix obtained from the data.
        :type neig: integer
        :param neig: number of dominant CSDM eigenvectors to annul from the data.
        :type data: numpy.ndarray
        :param data: the data which was used to calculate the CSDM. The projector is
            applied to it in order to cancel the strongest interferer.

        :return: numpy.ndarray
            csdm: the new cross-spectral-density matrix calculated from the data after
            the projector was applied to eliminate the strongest source.
        """

        # perform singular value decomposition to CSDM matrix
        u, s, vT = np.linalg.svd(CSDM)
        # chose only neig strongest eigenvectors
        u_m = u[:, :neig]   # columns are eigenvectors
        v_m = vT[:neig, :]  # rows (!) are eigenvectors
        # set-up projector
        proj = np.identity(CSDM.shape[0]) - np.dot(u_m, v_m)
        # apply projector to data - project largest eigenvectors out of data
        data = np.dot(proj, data)
        # calculate projected cross spectral density matrix
        csdm = np.dot(data, data.conj().T)
        return csdm
    
    def phase_matching(replica, K, processor):
        """
        Do phase matching of the replica vector with the CSDM matrix.
        :param replica: 2-D array containing the replica vectors of all parameter
            combinations (dim: [n_stats, n_param])
        :param K: 2-D array CSDM matrix (dim: [n_stats, n_stats])
        :param processor: Processor used for phase matching. bartlett or adaptive.
        """
        # calcualte inverse of CSDM matrix for adaptive processor
        if processor == "adaptive":
            K = np.linalg.inv(K)

        # reshape K matrix (or inverse of K) and append copy of it n_param times
        # along third dimension
        n_stats, n_param = replica.shape
        K = np.reshape(K, (n_stats, n_stats, 1))
        K = np.tile(K, (1, 1, n_param))

        # bartlett processor
        if processor == "bartlett":
            # initialize array for dot product
            dot1 = np.zeros((n_stats, n_param), dtype=complex)
            # first dot product - replica.conj().T with K
            for i in range(n_stats):
                dot1[i] = np.sum(np.multiply(replica.conj(), K[:,i,:]), axis=0)
            # second dot product - dot1 with replica
            beam = abs(np.sum(np.multiply(dot1, replica), axis=0))

        # adaptive processor
        elif processor == "adaptive":
            # initialize array for dot product
            dot1 = np.zeros((n_stats, n_param), dtype=complex)
            # first dot product - replica.conj().T with K_inv
            for i in range(n_stats):
                dot1[i] = np.sum(np.multiply(replica.conj(), K[:,i,:]), axis=0)
            # second dot product - dot1 with replica
            dot2 = np.sum(np.multiply(dot1, replica), axis=0)
            beam = abs((1. + 0.j) / dot2)

        return beam

    def matchedfield_beamformer(data, scoord, xrng, yrng, zrng, dx, dy, dz, svrng, ds,
        slow, fmin, fmax, Fs, w_length, w_delay,  processor="bartlett", df=0.2,
        neig=0, norm=True):
        """
        This routine estimates the back azimuth and phase velocity of incoming waves
        based on the algorithm presented in Corciulo et al., 2012 (in Geophysics).
        Can also be used to focus the beam to a certain coordinate, which must be
        specified with xmax, ymax, zmax. In this case, dx, dy, and dz need to be set
        to zero!
    
        :type data: numpy.ndarray
        :param data: time series of used stations (dim: [number of samples, number of stations])
        :type scoord: numpy.ndarray
        :param scoord: UTM coordinates of stations (dim: [number of stations, 2])
        :type xrng, yrng, zrng: tuple
        :param xrng, yrng, zrng: parameters for spatial grid search. Grid ranges
            from xrng[0] to xrng[1], yrng[0] to yrng[1], and zrng[0] to zrng[1].
        :type dx, dy, dz: float
        :param dx, dy, dz: grid resolution; increment from xrng[0] to xrng[1],
            yrng[0] to yrng[1], zrng[0] to zrng[1]
        :type svrng: tuple
        :param svrng: slowness interval used to calculate replica vector
        :type ds: float
        :param ds: slowness step used to calculate replica vector
        :type slow: boolean 
        :param slow: if true, svmin, svmax, dsv are slowness values. if false, velocity values
        :type fmin, fmax: float
        :param fmin, fmax: frequency range for which the beamforming result is calculated
        :type Fs: float
        :param Fs: sampling rate of data streams
        :type w_length: float
        :param w_length: length of sliding window in seconds. result is "averaged" over windows
        :type w_delay: float
        :param w_delay: delay of sliding window in seconds with respect to previous window
        :type processor: string
        :param processor: processor used to match the cross-spectral-density matrix to the
            replica vecotr. see Corciulo et al., 2012
        :type df: float
        :param df: frequency step between fmin and fmax
        :type neig: integer
        :param neig: number of dominant CSDM eigenvectors to annul from the data.
            enables to suppress strong sources.
        :type norm: boolean
        :param norm: if True (default), beam power is normalized

        :return: four numpy arrays:
            xcoord: grid coordinates in x-direction (dim: [number x-grid points, 1])
            ycoord: grid coordinates in y-direction (dim: [number y-grid points, 1])
            c: phase velocity (dim: [number of cs, 1])
            beamformer (dim: [number y-grid points, number x-grid points, number cs])
        """

        # number of stations
        n_stats = data.shape[1]

        # grid for search over location
        # if beam is fixed to a coordinate in x, y, or z
        if yrng[0] == yrng[1]:
            ycoord = np.array([yrng[0]])
        # if beam is calculated for a regular grid
        else:
            ycoord = np.arange(yrng[0], yrng[1] + dy, dy)
        # same for x ... 
        if xrng[0] == xrng[1]:
            xcoord = np.array([xrng[0]])
        else:
            xcoord = np.arange(xrng[0], xrng[1] + dx, dx)
        # and for z 
        if zrng[0] == zrng[1]:
            zcoord = np.array([zrng[0]])
        else:
            zcoord = np.arange(zrng[0], zrng[1] + dz, dz)
        # create meshgrids
        ygrid, xgrid = np.meshgrid(ycoord, xcoord)
        zgrid = np.zeros(xgrid.shape)
        ygrid = ygrid.reshape(ygrid.size)
        xgrid = xgrid.reshape(xgrid.size)
        zgrid = zgrid.reshape(zgrid.size)
        if zcoord.size > 1:
            ygrid = np.tile(ygrid, zcoord.size)
            xgrid = np.tile(xgrid, zcoord.size)
            zgrid_ = np.copy(zgrid)
            for i in range(zcoord.size - 1):
                zgrid = np.concatenate((zgrid, zgrid_ + zcoord[i+1]))

        # grid for search over slowness
        if svrng[0] == svrng[1]:
            s = np.array([svrng[0]]) / 1000.
        else:
            s = np.arange(svrng[0], svrng[1] + ds, ds) / 1000.
        if not slow:
            s = 1. / (s * 1.e6)
        # extend coordinate grids and slowness grid
        sgrid = np.zeros(xgrid.size) + s[0]
        ssize = sgrid.size
        if s.size > 1:
            ygrid = np.tile(ygrid, s.size)
            xgrid = np.tile(xgrid, s.size)
            zgrid = np.tile(zgrid, s.size)
            for i in range(s.size - 1):
                sgrid = np.concatenate((sgrid, np.zeros(ssize) + s[i+1]))
        # reshape for efficient calculation
        xscoord = np.tile(scoord[:,0].reshape(n_stats, 1), (1, xgrid.size))
        yscoord = np.tile(scoord[:,1].reshape(n_stats, 1), (1, ygrid.size))
        ygrid = np.tile(ygrid, (n_stats, 1))
        xgrid = np.tile(xgrid, (n_stats, 1))
        zgrid = np.tile(zgrid, (n_stats, 1))
        sgrid = np.tile(sgrid, (n_stats, 1))
        # number of parameter combinations
        n_param = xgrid.shape[1]

        # extract number of data points
        npts = data[:, 1].size
        # construct analysis frequencies
        freq = np.arange(fmin, fmax + df, df)
        # construct time vector for sliding window 
        w_time = np.arange(0, w_length, 1./Fs)
        npts_win = w_time.size
        npts_delay = int(w_delay * Fs)
        # number of analysis windows ('shots')
        nshots = int(np.floor((npts - w_time.size) / npts_delay)) + 1

        # initialize data steering vector:
        # dim: [number of frequencies, number of stations, number of analysis windows]
        vect_data = np.zeros((freq.size, n_stats, nshots), dtype=np.complex)

        # construct matrix for DFT calculation
        # dim: [number w_time points, number frequencies]
        matrice_int = np.exp(2. * np.pi * 1j * np.dot(w_time[:, None], freq[:, None].T))

        # initialize array for beamformer 
        beamformer = np.zeros(n_param)

        # calculate DFTs 
        for ii in range(n_stats):
            toto = data[:, ii]
            # now loop over shots
            n = 0
            while (n * npts_delay + npts_win) <= npts:
                # calculate DFT
                # dim: [number frequencies]
                adjust = np.dot(toto[n*npts_delay: n*npts_delay+npts_win][:, None],
                                np.ones((1, freq.size)))
                # mean averages over time axis
                data_freq = np.mean(np.multiply(adjust, matrice_int), axis=0)
                # fill data steering vector: ii'th station, n'th shot.
                # normalize in order not to bias strongest seismogram.
                # dim: [number frequencies, number stations, number shots]
                vect_data[:, ii, n] = (data_freq / abs(data_freq)).conj().T
                n += 1


        # loop over frequencies and perform beamforming
        for ll in range(freq.size):
            # calculate cross-spectral density matrix
            # dim: [number of stations X number of stations]
            K = calculate_CSDM(vect_data[ll,:,:], neig, norm)

            # calculate replica vector
            replica = np.exp(-1j * np.sqrt((xscoord - xgrid)**2 \
                + (yscoord - ygrid)**2 + zgrid**2) * 2. * np.pi * freq[ll] * sgrid)
            replica /= np.linalg.norm(replica, axis=0)
            replica = np.reshape(replica, (n_stats, n_param))

            # do phase matching
            beamformer += phase_matching(replica, K, processor)

        # normalize beamformer and reshape
        beamformer /= freq.size
        beamformer = np.reshape(beamformer, (ycoord.size, xcoord.size,
            zcoord.size, s.size), order="F")
        return ycoord, xcoord, zcoord, s*1000., beamformer
    
    #########################################################################      
    # END of matched-field beamforming functions
    #########################################################################

    def execute(self, stream, process_limits = None, origin_resource = None, **kwargs):
        '''
        '''
        if self.process_params is None:
            params_filename = self.pref_manager.get_value('params_filename')
            self.logger.info('Loading parameters from file %s.',
                             params_filename)
            with open(params_filename, 'r') as json_file:
                self.process_params = json.load(json_file)

            self.logger.info('Loaded parameters from file: %s.', self.process_params)
            

        start_time = process_limits[0]
        end_time = process_limits[1]

        self.logger.info('Stream passed to the execute function: %s', stream)

        for cur_trace in stream:
            self.logger.info('###Processing trace with id %s.', cur_trace.id)

            cur_scnl = (cur_trace.stats.station, cur_trace.stats.channel,
                        cur_trace.stats.network, cur_trace.stats.location)

            # Check if a result has to be created.
            self.check_result_needed(cur_scnl, start_time)

            # Get the channel instance from the inventory.
            cur_channel = self.project.geometry_inventory.get_channel(station = cur_trace.stats.station,
                                                                      name = cur_trace.stats.channel,
                                                                      network = cur_trace.stats.network,
                                                                      location = cur_trace.stats.location)
            if len(cur_channel) == 0:
                self.logger.error("No channel found for trace %s", cur_trace.id)
                continue
            elif len(cur_channel) > 1:
                self.logger.error("Multiple channels found for trace %s; channels: %s", cur_trace.id, cur_channel)
            else:
                cur_channel = cur_channel[0]

            beam_data = {}

            if cur_trace:
                self.logger.info('Computing MFP beamforming.')
                self.logger.info('cur_trace: %s', cur_trace)
                # Call the mfp_beamforming function here.
                
                
                
                
                
                
                
                
                
                
                
                # Assign the results in e.g. beam_data.


                # Pass the beam_data results to a function to save the results
                # in an instance variable for later use.
                # Store the beam data in the beam dictionary.
                self.save_beam_data(beam_data = beam_data,
                                    origin_resource = origin_resource)


    def save_beam_data(self, beam_data,
                       origin_resource = None):
        ''' Save the beam data in the instance.
        '''
        if not beam_data:
            return

        scnl = beam_data['scnl']
        start_time = beam_data['start_time']

        if scnl not in iter(self.beam_data.keys()):
            self.beam_data[scnl] = {}

        if scnl not in iter(self.save_day.keys()):
            self.save_day[scnl] = None

        if self.save_day[scnl] is None:
            self.save_day[scnl] = UTCDateTime(start_time.timestamp - start_time.timestamp % self.save_interval)

        self.beam_data[scnl][start_time.isoformat()] = beam_data



    def check_result_needed(self, scnl, start_time,
                            origin_resource = None):
        ''' Check if a result has to be created for the given SCNL.
        '''
        if scnl not in self.save_day.keys():
            return

        last_save_day = self.save_day[scnl]

        # Create a result if the current start time extends the save interval.
        # TODO: The naming of the result when saved by the time window looper
        # uses the wrong start- and end-times. Add a support to specify the
        # valid timespan of the result in the result instance.
        if start_time - last_save_day >= self.save_interval:
            self.create_result(scnl, origin_resource = origin_resource)
            self.save_day[scnl] = UTCDateTime(start_time.timestamp - start_time.timestamp % self.save_interval)



    def create_result(self, scnl, origin_resource = None):
        ''' Write the beam data for the given scnl to file.
        '''
        export_data = self.beam_data[scnl]

        first_time = UTCDateTime(sorted(export_data.keys())[0])
        last_time = UTCDateTime(sorted(export_data.keys())[-1])
        #last_key = sorted(export_data.iterkeys())[-1]
        #last_time = export_data[last_key]['end_time']
        #first_time = sorted([x['start_time'] for x in export_data.values()])[0]
        #last_time = sorted([x['end_time'] for x in export_data.values()])[-1]

        shelve_result = result.ShelveResult(name = 'beam',
                                            start_time = first_time,
                                            end_time = last_time,
                                            origin_name = self.name,
                                            origin_resource = origin_resource,
                                            sub_directory = (scnl[0],
                                                             scnl[1],
                                                             "{0:04d}_{1:03d}".format(first_time.year,
                                                                                      first_time.julday)),
                                            postfix = '_'.join(scnl),
                                            db = export_data)
        self.result_bag.add(shelve_result)
        self.logger.info("Published the result for scnl %s (%s to %s).", scnl,
                                                                         first_time.isoformat(),
                                                                         last_time.isoformat())

        self.beam_data[scnl] = {}


    def cleanup(self, origin_resource = None):
        ''' Publish all remaining psd data to results.
        '''
        for cur_scnl, cur_data in self.beam_data.items():
            if cur_data:
                self.create_result(cur_scnl, origin_resource = origin_resource)



