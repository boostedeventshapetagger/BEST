#=========================================================================================
# run_WW.py ------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Authors: Brendan Regnery, Reyer Band, Johan Bonilla ------------------------------------
#-----------------------------------------------------------------------------------------

#=========================================================================================
# Load Modules and Settings --------------------------------------------------------------
#=========================================================================================

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
from Configuration.AlCa.GlobalTag import GlobalTag


GT = '102X_mcRun2_asymptotic_v7'
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
        '/store/mc/RunIISummer16MiniAODv3/ZprimeToWW_narrow_M-2000_13TeV-madgraph/MINIAODSIM/PUMoriond17_94X_mcRun2_asymptotic_v3-v1/20000/A260D96B-9B03-EA11-BFE3-00259090765E.root'
        )
                            )
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#=========================================================================================
# Prepare and run producer ---------------------------------------------------------------
#=========================================================================================

# Apply a preselction
process.selectedAK8Jets = cms.EDFilter('PATJetSelector',
                                       src = cms.InputTag('slimmedJetsAK8'),
                                       cut = cms.string('500.0 < pt && pt < 3500.0 && abs(eta) < 2.4'),
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
        jetType = cms.string('W'),
        storeDaughters = cms.bool(True)
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

# Organize the running procedure
process.p = cms.Path(process.selectedAK8Jets*process.countAK8Jets*process.run)

