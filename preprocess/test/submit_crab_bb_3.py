from CRABClient.UserUtilities import config, getUsernameFromSiteDB
config = config()

config.General.requestName = 'bb_BEST_input_3'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.General.transferLogs = False

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'run_BB.py'
config.JobType.outputFiles = ['preprocess_BEST_bb.root']
config.JobType.allowUndistributedCMSSW = True 

config.Data.inputDataset = '/RSGravitonToBBbar_kMpl-01_M-7000_TuneCUEP8M1_13TeV-pythia8/RunIISummer16MiniAODv3-PUMoriond17_94X_mcRun2_asymptotic_v3-v1/MINIAODSIM'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.publication = False

config.Site.storageSite = 'T3_US_FNALLPC'
