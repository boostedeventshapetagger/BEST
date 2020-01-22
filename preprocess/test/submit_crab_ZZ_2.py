from CRABClient.UserUtilities import config, getUsernameFromSiteDB
config = config()

config.General.requestName = 'ZZ_BEST_input_2'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.General.transferLogs = False

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'run_ZZ.py'
config.JobType.outputFiles = ['preprocess_BEST_ZZ.root']
config.JobType.allowUndistributedCMSSW = True 

config.Data.inputDataset = '/RSGravToZZ_width0p1_M-3000_TuneCUETP8M1_13TeV-madgraph-pythia8/RunIISummer16MiniAODv3-PUMoriond17_94X_mcRun2_asymptotic_v3-v2/MINIAODSIM'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.publication = False

config.Site.storageSite = 'T3_US_FNALLPC'
