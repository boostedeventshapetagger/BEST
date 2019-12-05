from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")
config.General.requestName = 'RSGravZZ_1200GeV_trees'
config.General.workArea = 'NNoutputs'
config.General.transferLogs = True

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'run_ZZ.py'
config.JobType.inputFiles = ['TMVARegression_MLP.weights.xml']
config.JobType.outputFiles = ['histo_BESTprod.root']
config.JobType.allowUndistributedCMSSW = True

config.section_("Data")
config.Data.inputDataset = '/RSGravToZZ_width0p1_M-1200_TuneCUETP8M1_13TeV-madgraph-pythia8/RunIISummer16MiniAODv3-PUMoriond17_94X_mcRun2_asymptotic_v3-v1/MINIAODSIM'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.ignoreLocality = True
config.Data.publication = False
# This string is used to construct the output dataset name

config.section_("Site")
config.Site.storageSite = 'T3_US_FNALLPC'
config.Site.whitelist = ['T2_US_*']
