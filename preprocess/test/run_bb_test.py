#=========================================================================================
#run_QCD_test.py -------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Authors : Abhishek Das, Brendan Regnery, Reyer Band ------------------------------------
#-----------------------------------------------------------------------------------------

##############################################################################
#                    Load modules and Settings                               #
##############################################################################
import sys
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as opts
import copy
import os
from os import environ
from Configuration.AlCa.GlobalTag import GlobalTag

GT1 = '94X_mc2017_realistic_v17'
GT2 = '102X_mc2017_realistic_v7'

##############################################################################
#                     Start the Sequence                                     #
##############################################################################

process = cms.Process("run")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load("JetMETCorrections.Configuration.JetCorrectionServices_cff")
process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('JetMETCorrections.Configuration.JetCorrectors_cff')
process.load('JetMETCorrections.Configuration.CorrectedJetProducers_cff')
process.load('JetMETCorrections.Configuration.CorrectedJetProducersDefault_cff')

process.GlobalTag = GlobalTag(process.GlobalTag, GT2)


##############################################################################
#                     Files and Outputs                                      #
##############################################################################

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))
fileNames = cms.untracked.vstring(['root://cms-xrd-global.cern.ch//store/mc/RunIIFall17MiniAODv2/ZprimeToBB_narrow_M-2000_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/270000/58BD4FEB-D7D2-E911-9DB4-00259073E40A.root'])

process.source = cms.Source("PoolSource",
    fileNames = file_names)

process.MessageLogger.cerr.FwkReport.reportEvery = 10000
process.TFileService = cms.Service("TFileService", fileName = cms.string("BESTInputs_QCDSample.root") )
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
process.dump = cms.EDAnalyzer('EventContentAnalyzer')

##################################################################################
#                             Jet Correction                                     #
##################################################################################

from PhysicsTools.PatAlgos.tools.jetTools import *
from RecoBTag.MXNet.pfDeepBoostedJet_cff import *
#from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import * #runMetCorAndUncFromMiniAOD

updateJetCollection(
    process,
    jetSource = cms.InputTag('slimmedJetsAK8'),
    pvSource  = cms.InputTag('offlineSlimmedPrimaryVertices'),
    svSource  = cms.InputTag('slimmedSecondaryVertices'),
    rParam    = 0.8,
    jetCorrections = ('AK8PFchs', cms.vstring(['L2Relative', 'L3Absolute', 'L2L3Residual']), 'None'),
    postfix        = '',
    printWarning   = False
)

process.JECSequence = cms.Sequence(
    process.patJetCorrFactors   *
    process.updatedPatJets      *
    #process.patJetCorrFactorsTransientCorrected *                                                                                           
    #process.updatedPatJetsTransientCorrected    *                                                                                           
    process.selectedUpdatedPatJets
    )

################################################################################
#                            Jet Smearing                                      #
################################################################################
#see https://github.com/cms-sw/cmssw/blob/CMSSW_10_2_X/PhysicsTools/PatUtils/python/patPFMETCorrections_cff.py#L106
from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
from RecoMET.METProducers.METSigParams_cfi import *
JetSmearing = cms.EDProducer("SmearedPATJetProducer",
    src             = cms.InputTag("slimmedJets"),
    enabled         = cms.bool(True),   # If False, no smearing is performed                                                                 
    rho             = cms.InputTag("fixedGridRhoFastjetAll"),
    skipGenMatching = cms.bool(False),  # If True, always skip gen jet matching and smear jet with a random gaussian                         
    algopt          = cms.string('AK8PFchs_pt'),
    algo            = cms.string('AK8PFchs'),
    genJets         = cms.InputTag("slimmedGenJets"),
    dRMax           = cms.double(0.2),  # = cone size (0.4) / 2                                                                              
    dPtMaxFactor    = cms.double(3),    # dPt < 3 * resolution                                                                               
    variation       = cms.int32(0),     # If not specified, default to 0                                                                     
    seed            = cms.uint32(37428479), # If not specified, default to 37428479                                                          
    debug           = cms.untracked.bool(False)
)

process.mySmearedAK8Jets = JetSmearing.clone(
    src       = cms.InputTag("selectedUpdatedPatJets"),
    algopt    = cms.string('AK8PFPuppi_pt'),
    algo      = cms.string('AK8PFPuppi'),
    genJets   = cms.InputTag("slimmedGenJetsAK8"),
    dRMax     = cms.double(0.4),  # = cone size (0.8) / 2                                                                                    
    variation = cms.int32(0)
)

process.mySmearedAK8JetsUp = process.mySmearedAK8Jets.clone(
    variation = cms.int32(+1)
)
process.mySmearedAK8JetsDown = process.mySmearedAK8Jets.clone(
    variation = cms.int32(-1)
)


process.SmearedJetSequence = cms.Sequence(
    process.mySmearedAK8Jets     *
    process.mySmearedAK8JetsUp   *
    process.mySmearedAK8JetsDown
)

################################################################################
#                            BESTProducer                                      #
################################################################################

process.selectedAK8Jets = cms.EDFilter('PATJetSelector',
                                       src = cms.InputTag('mySmearedAK8Jets'),
                                       cut = cms.string('pt > 500.0 && abs(eta) < 2.4'),
                                       filter = cms.bool(True)
                                       )

process.countAK8Jets = cms.EDFilter("PATCandViewCountFilter",
                                    minNumber = cms.uint32(1),
                                    maxNumber = cms.uint32(99999),
                                    src = cms.InputTag('mySmearedAK8Jets')
#                                    filter = cms.bool(True)
                                    )
# Run the producer
process.run = cms.EDProducer('BESTProducer',
                             inputJetColl = cms.string('mySmearedAK8Jets'),
        jetColl = cms.string('PUPPI'),                     
        jetType = cms.string('b'),
        storeDaughters = cms.bool(True)
)

process.p = cms.Path(process.JECSequence *
                     process.SmearedJetSequence *
                     process.selectedAK8Jets *
                     process.countAK8Jets *
                     process.run)



#Z type dataset - /store/mc/RunIIFall17MiniAODv2/RadionToZZ_narrow_M-5000_TuneCP5_13TeV-madgraph/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/240000/88C32EBF-A689-E911-BE4D-A4BF0112BCD4.root

#B type dataset - /store/mc/RunIIFall17MiniAODv2/ZprimeToBB_narrow_M-2000_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/270000/58BD4FEB-D7D2-E911-9DB4-00259073E40A.root
#'b'
