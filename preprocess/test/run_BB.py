#=========================================================================================
# run_BB.py ------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Authors: Brendan Regnery, Reyer Band ---------------------------------------------------
#-----------------------------------------------------------------------------------------

#=========================================================================================
# Load Modules and Settings --------------------------------------------------------------
#=========================================================================================

import FWCore.ParameterSet.Config as cms
from JMEAnalysis.JetToolbox.jetToolbox_cff import jetToolbox
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
from Configuration.AlCa.GlobalTag import GlobalTag

process = cms.Process("run")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("JetMETCorrections.Configuration.JetCorrectionServices_cff")
process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load("RecoBTag.Configuration.RecoBTag_cff")

process.GlobalTag = GlobalTag(process.GlobalTag, '80X_mcRun2_asymptotic_v4')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1),
                                        allowUnscheduled = cms.untracked.bool(True))

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        # This single file can be used for testing
        #'root://cmsxrootd-site.fnal.gov//store/mc/RunIISummer16MiniAODv3/RSGravitonToBBbar_kMpl-01_M-1000_TuneCUEP8M1_13TeV-pythia8/MINIAODSIM/PUMoriond17_94X_mcRun2_asymptotic_v3-v1/110000/221B5B32-8CD2-E811-8620-24BE05CEEB61.root'
	)
)
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#=========================================================================================
# Remake the Jet Collections -------------------------------------------------------------
#=========================================================================================

# Adjust the jet collection to include tau4
jetToolbox( process, 'ak8', 'jetsequence', 'out',
    updateCollection = 'slimmedJetsAK8',
    JETCorrPayload= 'AK8PFchs',
    addNsub = True,
    maxTau = 4
)

#=========================================================================================
# Prepare and run producer ---------------------------------------------------------------
#=========================================================================================

# Apply a preselction
process.selectedAK8Jets = cms.EDFilter('PATJetSelector',
    src = cms.InputTag('selectedPatJetsAK8PFCHS'),
    cut = cms.string('pt > 100.0 && abs(eta) < 2.4'),
    filter = cms.bool(True)
)

process.countAK8Jets = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(99999),
    src = cms.InputTag("selectedAK8Jets"),
    filter = cms.bool(True)
)

# Run the producer
process.run = cms.EDProducer('BESTProducer',
	inputJetColl = cms.string('selectedAK8Jets'),
        jetType = cms.string('b'),
        jetColl = cms.string('CHS')
)

process.TFileService = cms.Service("TFileService", fileName = cms.string("preprocess_BEST_bb.root") )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("ana_out.root"),
                               #SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      #'keep *_*AK8*_*_*', #'drop *',
                                                                      'keep *_*run*_*_*'
                                                                      #, 'keep *_goodPatJetsCATopTagPF_*_*'
                                                                      #, 'keep recoPFJets_*_*_*'
                                                                      ) 
                               )
process.outpath = cms.EndPath(process.out)

# Organize the running process
process.p = cms.Path(
    process.selectedAK8Jets
    * process.countAK8Jets
    * process.run
)
