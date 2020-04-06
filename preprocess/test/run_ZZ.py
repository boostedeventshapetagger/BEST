#=========================================================================================
# run_ZZ.py ------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Authors: Brendan Regnery, Reyer Band ---------------------------------------------------
#-----------------------------------------------------------------------------------------

#=========================================================================================
# Load Modules and Settings --------------------------------------------------------------
#=========================================================================================

import FWCore.ParameterSet.Config as cms
#from JMEAnalysis.JetToolbox.jetToolbox_cff import jetToolbox
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
from Configuration.AlCa.GlobalTag import GlobalTag


GT = '102X_mc2017_realistic_v7'
process = cms.Process("run")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load("JetMETCorrections.Configuration.JetCorrectionServices_cff")
process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")
process.GlobalTag = GlobalTag(process.GlobalTag, GT)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))


process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        '/store/mc/RunIIFall17MiniAODv2/RadionToZZ_narrow_M-5000_TuneCP5_13TeV-madgraph/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/240000/88C32EBF-A689-E911-BE4D-A4BF0112BCD4.root'
#        'file://88C32EBF-A689-E911-BE4D-A4BF0112BCD4.root'
        )
                            )
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#=========================================================================================
# Remake the Jet Collections -------------------------------------------------------------
#=========================================================================================

# jetToolbox( process, 'ak8', 'jetsequence', 'out',
#     updateCollection = 'slimmedJetsAK8',
#     JETCorrPayload= 'AK8PFPuppi',
#     PUMethod='Puppi',
#     runOnMC=True,    
# #    JETCorrPayload= 'AK8PFchs',
#     addNsub = True,
#     maxTau = 4
# )

#=========================================================================================
# Prepare and run producer ---------------------------------------------------------------
#=========================================================================================

# Apply a preselction
process.selectedAK8Jets = cms.EDFilter('PATJetSelector',
                                       src = cms.InputTag('slimmedJetsAK8'),
                                       cut = cms.string('pt > 500.0 && abs(eta) < 2.4'),
                                       filter = cms.bool(True)
                                       )

process.countAK8Jets = cms.EDFilter("PATCandViewCountFilter",
                                    minNumber = cms.uint32(1),
                                    maxNumber = cms.uint32(99999),
                                    src = cms.InputTag('slimmedJetsAK8')
#                                    filter = cms.bool(True)
                                    )


# Run the producer
process.run = cms.EDProducer('BESTProducer',
	inputJetColl = cms.string('slimmedJetsAK8'),
        jetColl = cms.string('PUPPI'),                     
        jetType = cms.string('Z')
#	pdgIDforMatch = cms.int32(23),
#	NNtargetX = cms.int32(1),
#	NNtargetY = cms.int32(1),
#	isMC = cms.int32(1),
#        isQCD = cms.int32(0),
#	doMatch = cms.int32(0),
#	usePuppi = cms.int32(0)

)
process.TFileService = cms.Service("TFileService", fileName = cms.string("BESTInputs.root") )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("ana_out.root"),
                               SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               outputCommands = cms.untracked.vstring('drop *',
								      'keep *_fixedGridRhoAll_*_*',
                                                                      'keep *_run_*_*',
                                                                      #, 'keep *_goodPatJetsCATopTagPF_*_*'
                                                                      #, 'keep recoPFJets_*_*_*'
                                                                      ) 
                               )
process.outpath = cms.EndPath(process.out)

# Organize the running process
process.p = cms.Path(process.selectedAK8Jets*process.countAK8Jets*process.run)

