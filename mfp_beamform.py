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



