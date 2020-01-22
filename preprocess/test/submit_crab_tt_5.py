from CRABClient.UserUtilities import config, getUsernameFromSiteDB
config = config()

config.General.requestName = 'tt_BEST_input_5'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.General.transferLogs = False

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'run_tt.py'
config.JobType.outputFiles = ['preprocess_BEST_tt.root']
config.JobType.allowUndistributedCMSSW = True 

config.Data.inputDataset = '/ZprimeToTT_M6000_W60_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/MINIAODSIM'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.publication = False

config.Site.storageSite = 'T3_US_FNALLPC'
