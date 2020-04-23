// -*- C++ -*-
//========================================================================================
// Package:    BEST/preprocess                  ---------------------------------------
// Class:      BESTProducer                     ---------------------------------------
//----------------------------------------------------------------------------------------
/**\class BESTProducer BESTProducer.cc BEST/preprocess/plugins/BESTProducer.cc
------------------------------------------------------------------------------------------
 Description: This class preprocesses MC samples so that they can be used with BEST ---
 -----------------------------------------------------------------------------------------
 Implementation:                                                                       ---
     This EDProducer is meant to be used with CMSSW_9_4_8                              ---
*/
//========================================================================================
// Authors:  Brendan Regnery, Justin Pilot, Reyer Band, Devin Taylor ---------------------
//         Created:  WED, 8 Aug 2018 21:00:28 GMT  ---------------------------------------
//========================================================================================
//////////////////////////////////////////////////////////////////////////////////////////


// system include files
#include <memory>
#include <thread>
#include <iostream>

// FWCore include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// Data Formats and tools include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
//#include "DataFormats/VertexReco/interface/VertexFwd.h"
//#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "PhysicsTools/CandUtils/interface/EventShapeVariables.h"
#include "PhysicsTools/CandUtils/interface/Thrust.h"

// Fast Jet Include files
#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>
#include "fastjet/tools/Filter.hh"
#include <fastjet/ClusterSequence.hh>
#include <fastjet/ActiveAreaSpec.hh>
#include <fastjet/ClusterSequenceArea.hh>

// ROOT include files
#include "TTree.h"
#include "TFile.h"
#include "TH2F.h"
#include "TLorentzVector.h"
#include "TCanvas.h"

// user made files
#include "BESTtoolbox.h"

///////////////////////////////////////////////////////////////////////////////////
// Define a namespace -------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

namespace best {

    // enumerate possible jet types
    enum JetType { Q, H, t, W, Z, b};

    // enumerate possible jet collections
    enum JetColl{ CHS, PUPPI};

    // create a struct to help with mapping string label to enum value
    struct JetTypeStringToEnum {
        const char label;
        JetType value;
    };

    // create a struct to help with mapping string label to enum value
    struct JetCollStringToEnum {
        const char* label;
        JetColl value;
    };

    // Create a mapping from the input jetType to enum
    JetType jetTypeFromString(const std::string& label) {
        static const JetTypeStringToEnum jetTypeStringToEnumMap[] = {
            {'Q', Q},
	    {'H', H},
            {'t', t},
            {'W', W},
            {'Z', Z},
            {'b', b}
        };

        JetType value = (JetType)-1;
        bool found = false;
        for (int i = 0; jetTypeStringToEnumMap[i].label && (!found); ++i){
            if (!strcmp(label.c_str(), &jetTypeStringToEnumMap[i].label) ) {
                found = true;
                value = jetTypeStringToEnumMap[i].value;
            }
        }

        // Throw an error if user inputs an unrecognized type
        if (!found){
            throw cms::Exception("JetTypeError") << label << " is not a recognized JetType";
        }

        return value;
    }


    // Create a mapping from the input jetColl to enum
    JetColl jetCollFromString(const std::string& label) {
        static const JetCollStringToEnum jetCollStringToEnumMap[] = {
            {"CHS", CHS},
            {"PUPPI", PUPPI}
        };

        JetColl value = (JetColl)-1;
        bool found = false;
        for (int i = 0; jetCollStringToEnumMap[i].label && (!found); ++i){
            if (!strcmp(label.c_str(), jetCollStringToEnumMap[i].label) ) {
                found = true;
                value = jetCollStringToEnumMap[i].value;
            }
        }

        // Throw an error if user inputs an unrecognized type
        if (!found){
            throw cms::Exception("JetCollError") << label << " is not a recognized JetColl";
        }

        return value;
    }
}


///////////////////////////////////////////////////////////////////////////////////
// Class declaration --------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

class BESTProducer : public edm::stream::EDProducer<> {
   public:
      explicit BESTProducer(const edm::ParameterSet&);
      ~BESTProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      //===========================================================================
      // User functions -----------------------------------------------------------
      //===========================================================================

   private:
      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

      //===========================================================================
      // Member Data --------------------------------------------------------------
      //===========================================================================

      // Input variables
      std::string inputJetColl_;
      best::JetType jetType_;
      best::JetColl jetColl_;
      bool storeDaughters;

      // Tree variables
      TTree *jetTree;
      std::map<std::string, float> treeVars;
      std::vector<std::string> listOfVars;
      std::map<std::string, std::vector<float> > jetVecVars;
      std::vector<std::string> listOfVecVars;
      std::map<std::string, std::array<std::array<std::array<float, 1>, 31>, 31> > imgVars;
      std::vector<std::string> listOfImgVars;

      // Tokens
      //edm::EDGetTokenT<std::vector<pat::PackedCandidate> > pfCandsToken_;
      edm::EDGetTokenT<std::vector<pat::Jet> > ak8JetsToken_;
      //edm::EDGetTokenT<std::vector<pat::Jet> > ak4JetsToken_;
      edm::EDGetTokenT<std::vector<reco::GenParticle> > genPartToken_;
      edm::EDGetTokenT<std::vector<reco::VertexCompositePtrCandidate> > secVerticesToken_;
      edm::EDGetTokenT<std::vector<reco::Vertex> > verticesToken_;

      //edm::EDGetTokenT<std::vector<pat::Jet> > ak8CHSSoftDropSubjetsToken_;

      //edm::EDGetTokenT<edm::TriggerResults> trigResultsToken_;
      //edm::EDGetTokenT<bool> BadChCandFilterToken_;
      //edm::EDGetTokenT<bool> BadPFMuonFilterToken_;
};

///////////////////////////////////////////////////////////////////////////////////
// constants, enums and typedefs --------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////
// static data member definitions -------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////
// Constructors -------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

BESTProducer::BESTProducer(const edm::ParameterSet& iConfig):
    inputJetColl_ (iConfig.getParameter<std::string>("inputJetColl")),
    jetType_ (best::jetTypeFromString(iConfig.getParameter<std::string>("jetType"))),
    jetColl_ (best::jetCollFromString(iConfig.getParameter<std::string>("jetColl"))),
    storeDaughters (iConfig.getParameter<bool>("storeDaughters"))
{

    //------------------------------------------------------------------------------
    // Prepare TFile Service -------------------------------------------------------
    //------------------------------------------------------------------------------

    edm::Service<TFileService> fs;
    jetTree = fs->make<TTree>("jetTree","jetTree");

    //------------------------------------------------------------------------------
    // Create tree variables and branches ------------------------------------------
    //------------------------------------------------------------------------------
    // listOfVars is the flat part of the TTree ------------------------------------
    // listOfVecVars is the vector part of the TTree -------------------------------
    //------------------------------------------------------------------------------

    // AK8 jet variables
    listOfVars.push_back("nJets");

    listOfVars.push_back("jetAK8_phi");
    listOfVars.push_back("jetAK8_eta");
    listOfVars.push_back("jetAK8_pt");
    listOfVars.push_back("jetAK8_mass");
    listOfVars.push_back("jetAK8_SoftDropMass");

    // Vertex Variables
    listOfVars.push_back("nSecondaryVertices");
    listOfVecVars.push_back("SV_pt");
    listOfVecVars.push_back("SV_eta");
    listOfVecVars.push_back("SV_phi");
    listOfVecVars.push_back("SV_mass");
    listOfVecVars.push_back("SV_nTracks");
    listOfVecVars.push_back("SV_chi2");
    listOfVecVars.push_back("SV_Ndof");

    // Deep Jet b Discriminants
    listOfVars.push_back("bDisc");
    listOfVars.push_back("bDisc1");
    listOfVars.push_back("bDisc2");

    // nsubjettiness
    listOfVars.push_back("jetAK8_Tau4");
    listOfVars.push_back("jetAK8_Tau3");
    listOfVars.push_back("jetAK8_Tau2");
    listOfVars.push_back("jetAK8_Tau1");
    listOfVars.push_back("jetAK8_Tau32");
    listOfVars.push_back("jetAK8_Tau21");

    // Fox Wolfram Moments
    listOfVars.push_back("FoxWolfH1_Higgs");
    listOfVars.push_back("FoxWolfH2_Higgs");
    listOfVars.push_back("FoxWolfH3_Higgs");
    listOfVars.push_back("FoxWolfH4_Higgs");

    listOfVars.push_back("FoxWolfH1_Top");
    listOfVars.push_back("FoxWolfH2_Top");
    listOfVars.push_back("FoxWolfH3_Top");
    listOfVars.push_back("FoxWolfH4_Top");

    listOfVars.push_back("FoxWolfH1_W");
    listOfVars.push_back("FoxWolfH2_W");
    listOfVars.push_back("FoxWolfH3_W");
    listOfVars.push_back("FoxWolfH4_W");

    listOfVars.push_back("FoxWolfH1_Z");
    listOfVars.push_back("FoxWolfH2_Z");
    listOfVars.push_back("FoxWolfH3_Z");
    listOfVars.push_back("FoxWolfH4_Z");

    // Event Shape Variables
    listOfVars.push_back("isotropy_Higgs");
    listOfVars.push_back("sphericity_Higgs");
    listOfVars.push_back("aplanarity_Higgs");
    listOfVars.push_back("thrust_Higgs");

    listOfVars.push_back("isotropy_Top");
    listOfVars.push_back("sphericity_Top");
    listOfVars.push_back("aplanarity_Top");
    listOfVars.push_back("thrust_Top");

    listOfVars.push_back("isotropy_W");
    listOfVars.push_back("sphericity_W");
    listOfVars.push_back("aplanarity_W");
    listOfVars.push_back("thrust_W");

    listOfVars.push_back("isotropy_Z");
    listOfVars.push_back("sphericity_Z");
    listOfVars.push_back("aplanarity_Z");
    listOfVars.push_back("thrust_Z");

    // Jet Mass
    listOfVars.push_back("nJets_Higgs");
    listOfVars.push_back("nJets_Top");
    listOfVars.push_back("nJets_W");
    listOfVars.push_back("nJets_Z");

    listOfVars.push_back("jet12_mass_Higgs");
    listOfVars.push_back("jet23_mass_Higgs");
    listOfVars.push_back("jet13_mass_Higgs");
    listOfVars.push_back("jet1234_mass_Higgs");

    listOfVars.push_back("jet12_mass_Top");
    listOfVars.push_back("jet23_mass_Top");
    listOfVars.push_back("jet13_mass_Top");
    listOfVars.push_back("jet1234_mass_Top");

    listOfVars.push_back("jet12_mass_W");
    listOfVars.push_back("jet23_mass_W");
    listOfVars.push_back("jet13_mass_W");
    listOfVars.push_back("jet1234_mass_W");

    listOfVars.push_back("jet12_mass_Z");
    listOfVars.push_back("jet23_mass_Z");
    listOfVars.push_back("jet13_mass_Z");
    listOfVars.push_back("jet1234_mass_Z");
    //Subjet CosTheta and delta CosTheta
    listOfVars.push_back("jet12_CosTheta_Higgs");
    listOfVars.push_back("jet23_CosTheta_Higgs");
    listOfVars.push_back("jet13_CosTheta_Higgs");
    listOfVars.push_back("jet1234_CosTheta_Higgs");

    listOfVars.push_back("jet12_CosTheta_Top");
    listOfVars.push_back("jet23_CosTheta_Top");
    listOfVars.push_back("jet13_CosTheta_Top");
    listOfVars.push_back("jet1234_CosTheta_Top");

    listOfVars.push_back("jet12_CosTheta_W");
    listOfVars.push_back("jet23_CosTheta_W");
    listOfVars.push_back("jet13_CosTheta_W");
    listOfVars.push_back("jet1234_CosTheta_W");

    listOfVars.push_back("jet12_CosTheta_Z");
    listOfVars.push_back("jet23_CosTheta_Z");
    listOfVars.push_back("jet13_CosTheta_Z");
    listOfVars.push_back("jet1234_CosTheta_Z");

    listOfVars.push_back("jet12_DeltaCosTheta_Higgs");
    listOfVars.push_back("jet13_DeltaCosTheta_Higgs");
    listOfVars.push_back("jet23_DeltaCosTheta_Higgs");

    listOfVars.push_back("jet12_DeltaCosTheta_Top");
    listOfVars.push_back("jet13_DeltaCosTheta_Top");
    listOfVars.push_back("jet23_DeltaCosTheta_Top");

    listOfVars.push_back("jet12_DeltaCosTheta_W");
    listOfVars.push_back("jet13_DeltaCosTheta_W");
    listOfVars.push_back("jet23_DeltaCosTheta_W");

    listOfVars.push_back("jet12_DeltaCosTheta_Z");
    listOfVars.push_back("jet13_DeltaCosTheta_Z");
    listOfVars.push_back("jet23_DeltaCosTheta_Z");

    // Jet Asymmetry
    listOfVars.push_back("asymmetry_Higgs");
    listOfVars.push_back("asymmetry_Top");
    listOfVars.push_back("asymmetry_W");
    listOfVars.push_back("asymmetry_Z");

    // add the daughter and rest frame information
    if(storeDaughters == true){

        // Jet PF Candidate Variables
        listOfVecVars.push_back("jet_PF_candidate_pt");
        listOfVecVars.push_back("jet_PF_candidate_phi");
        listOfVecVars.push_back("jet_PF_candidate_eta");

        listOfVecVars.push_back("HiggsFrame_PF_candidate_px");
        listOfVecVars.push_back("HiggsFrame_PF_candidate_py");
        listOfVecVars.push_back("HiggsFrame_PF_candidate_pz");
        listOfVecVars.push_back("HiggsFrame_PF_candidate_energy");

        listOfVecVars.push_back("TopFrame_PF_candidate_px");
        listOfVecVars.push_back("TopFrame_PF_candidate_py");
        listOfVecVars.push_back("TopFrame_PF_candidate_pz");
        listOfVecVars.push_back("TopFrame_PF_candidate_energy");

        listOfVecVars.push_back("WFrame_PF_candidate_px");
        listOfVecVars.push_back("WFrame_PF_candidate_py");
        listOfVecVars.push_back("WFrame_PF_candidate_pz");
        listOfVecVars.push_back("WFrame_PF_candidate_energy");

        listOfVecVars.push_back("ZFrame_PF_candidate_px");
        listOfVecVars.push_back("ZFrame_PF_candidate_py");
        listOfVecVars.push_back("ZFrame_PF_candidate_pz");
        listOfVecVars.push_back("ZFrame_PF_candidate_energy");

        // PUPPI weights
        listOfVecVars.push_back("PUPPI_Weights");

        // rest frame subjet variables
        listOfVecVars.push_back("HiggsFrame_jet_px");
        listOfVecVars.push_back("HiggsFrame_jet_py");
        listOfVecVars.push_back("HiggsFrame_jet_pz");
        listOfVecVars.push_back("HiggsFrame_jet_energy");

        listOfVecVars.push_back("TopFrame_jet_px");
        listOfVecVars.push_back("TopFrame_jet_py");
        listOfVecVars.push_back("TopFrame_jet_pz");
        listOfVecVars.push_back("TopFrame_jet_energy");

        listOfVecVars.push_back("WFrame_jet_px");
        listOfVecVars.push_back("WFrame_jet_py");
        listOfVecVars.push_back("WFrame_jet_pz");
        listOfVecVars.push_back("WFrame_jet_energy");

        listOfVecVars.push_back("ZFrame_jet_px");
        listOfVecVars.push_back("ZFrame_jet_py");
        listOfVecVars.push_back("ZFrame_jet_pz");
        listOfVecVars.push_back("ZFrame_jet_energy");
    }

    // rest frame jet image variables
    listOfImgVars.push_back("HiggsFrame_image");
    listOfImgVars.push_back("TopFrame_image");
    listOfImgVars.push_back("WFrame_image");
    listOfImgVars.push_back("ZFrame_image");

    // Make Branches for each variable
    for (unsigned i = 0; i < listOfVars.size(); i++){
        treeVars[ listOfVars[i] ] = -999.99;
        jetTree->Branch( (listOfVars[i]).c_str() , &(treeVars[ listOfVars[i] ]), (listOfVars[i]+"/F").c_str() );
    }

    // Make Branches for each of the jet constituents' variables
    for (unsigned i = 0; i < listOfVecVars.size(); i++){
        jetTree->Branch( (listOfVecVars[i]).c_str() , &(jetVecVars[ listOfVecVars[i] ]) );
    }

    // Make branches for each of the images
    for (unsigned i = 0; i < listOfImgVars.size(); i++){
        jetTree->Branch( (listOfImgVars[i]).c_str() , &(imgVars[ listOfImgVars[i] ]), (listOfImgVars[i]+"[31][31][1]/F").c_str() );
    }

    //------------------------------------------------------------------------------
    // Define input tags -----------------------------------------------------------
    //------------------------------------------------------------------------------

    // AK8 Jets
    edm::InputTag ak8JetsTag_;
    ak8JetsTag_ = edm::InputTag("slimmedJetsAK8", "", "PAT");
    //    ak8JetsTag_ = edm::InputTag(inputJetColl_, "", "run"); // this may be needed as an option for 2016 mc
    ak8JetsToken_ = consumes<std::vector<pat::Jet> >(ak8JetsTag_);

    // Gen Particles
    edm::InputTag genPartTag_;
    genPartTag_ = edm::InputTag("prunedGenParticles", "", "PAT");
    genPartToken_ = consumes<std::vector<reco::GenParticle> >(genPartTag_);

    // Primary Vertices
    edm::InputTag verticesTag_;
    verticesTag_ = edm::InputTag("offlineSlimmedPrimaryVertices", "", "PAT");
    verticesToken_ = consumes<std::vector<reco::Vertex> >(verticesTag_);

    // Secondary Vertices
    edm::InputTag secVerticesTag_;
    secVerticesTag_ = edm::InputTag("slimmedSecondaryVertices", "", "PAT");
    secVerticesToken_ = consumes<std::vector<reco::VertexCompositePtrCandidate> >(secVerticesTag_);
}

///////////////////////////////////////////////////////////////////////////////////
// Destructor ---------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

BESTProducer::~BESTProducer()
{

    // do anything that needs to be done at destruction time
    // (eg. close files, deallocate, resources etc.)

}

///////////////////////////////////////////////////////////////////////////////////
// Member Functions ---------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

//=================================================================================
// Method called for each event ---------------------------------------------------
//=================================================================================

void
BESTProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;
    using namespace fastjet;
    using namespace std;

    typedef reco::Candidate::PolarLorentzVector fourv;

    //------------------------------------------------------------------------------
    // Create miniAOD object collections -------------------------------------------
    //------------------------------------------------------------------------------

    // Find objects corresponding to the token and link to the handle
    Handle< std::vector<pat::Jet> > ak8JetsCollection;
    iEvent.getByToken(ak8JetsToken_, ak8JetsCollection);
    vector<pat::Jet> ak8Jets = *ak8JetsCollection.product();

    Handle< std::vector<reco::GenParticle> > genPartCollection;
    iEvent.getByToken(genPartToken_, genPartCollection);
    vector<reco::GenParticle> genPart = *genPartCollection.product();

    Handle< std::vector<reco::Vertex> > vertexCollection;
    iEvent.getByToken(verticesToken_, vertexCollection);
    vector<reco::Vertex> pVertices = *vertexCollection.product();

    Handle< std::vector<reco::VertexCompositePtrCandidate> > secVertexCollection;
    iEvent.getByToken(secVerticesToken_, secVertexCollection);
    vector<reco::VertexCompositePtrCandidate> secVertices = *secVertexCollection.product();

    //------------------------------------------------------------------------------
    // Gen Particles Loop ----------------------------------------------------------
    //------------------------------------------------------------------------------
    // This makes a TLorentz Vector for each generator Heavy Object to use for jet
    // matching
    //------------------------------------------------------------------------------
    // Please note that the Jet Type has been enumerated:
    // QCD -> 0, H -> 1, t -> 2, W -> 3, Z -> 4, b -> 5
    //------------------------------------------------------------------------------
    int pdgID = -99;
    switch(jetType_){
    case 1:
      pdgID = 25;
      break;
    case 2:
      pdgID = 6;
      break;
    case 3:
      pdgID =  24;
      break;
    case 4:
      pdgID = 23;
      break;
    case 5:
      pdgID = 5;
      break;
    default:
      pdgID = -99;
      break;
    }
    // Store heavy particle for jet matching
    std::vector<TLorentzVector> genParticleToMatch;
    if(jetType_ != 0){
        for (vector<reco::GenParticle>::const_iterator genBegin = genPart.begin(), genEnd = genPart.end(), ipart = genBegin; ipart != genEnd; ++ipart){
            if(abs(ipart->pdgId() ) == pdgID){
                genParticleToMatch.push_back( TLorentzVector(ipart->px(), ipart->py(), ipart->pz(), ipart->energy() ) );
            }
        }
    }

    //------------------------------------------------------------------------------
    // AK8 Jet Loop ----------------------------------------------------------------
    //------------------------------------------------------------------------------
    // This loop makes a tree entry for each jet of interest -----------------------
    //------------------------------------------------------------------------------

    for (vector<pat::Jet>::const_iterator jetBegin = ak8Jets.begin(), jetEnd = ak8Jets.end(), ijet = jetBegin; ijet != jetEnd; ++ijet){
        bool GenMatching = false;
        TLorentzVector jet(ijet->px(), ijet->py(), ijet->pz(), ijet->energy() );

        if(ijet->subjets("SoftDropPuppi").size() >=2 && ijet->numberOfDaughters() > 2 && ijet->pt() >= 500 && fabs(ijet->eta()) < 2.4 &&ijet->userFloat("ak8PFJetsPuppiSoftDropMass") > 10) {

            // gen particle loop, only relevant for non-QCD jets
            if (jetType_ !=0){
                for (size_t iGenParticle = 0; iGenParticle < genParticleToMatch.size(); iGenParticle++){
                    // Check if jet matches any saved genParticle
                    if(jet.DeltaR(genParticleToMatch[iGenParticle]) < 0.1){
                        GenMatching = true;
                    }
                }
            }
            if (GenMatching || (jetType_ == 0)){

                // Store Jet Variables
                treeVars["nJets"] = ak8Jets.size();
                storeJetVariables(treeVars, ijet, jetColl_);

                // Secondary Vertex Variables
                storeSecVertexVariables(treeVars, jetVecVars, jet, secVertices);

                // Create structures for storing daughters and rest frame jets
                vector<reco::Candidate * > daughtersOfJet;
                map<string, vector<TLorentzVector>* > boostedDaughters;
                map<string, vector<fastjet::PseudoJet> > restJets;

                // Get all of the Jet's daughters
                getJetDaughters(daughtersOfJet, ijet);
                if (daughtersOfJet.size() < 3) continue;

                // Higgs Rest Frame Variables
                calcBESvariables(treeVars, daughtersOfJet, boostedDaughters, ijet, restJets, imgVars, "Higgs", 125.);

                // Top Rest Frame Variables
                calcBESvariables(treeVars, daughtersOfJet, boostedDaughters, ijet, restJets, imgVars, "Top", 172.5);

                // W Rest Frame Variables
                calcBESvariables(treeVars, daughtersOfJet, boostedDaughters, ijet, restJets, imgVars, "W", 80.4);

                // Z Rest Frame Variables
                calcBESvariables(treeVars, daughtersOfJet, boostedDaughters, ijet, restJets, imgVars, "Z", 91.2);

                // store daughters, rest frame daughters, and rest frame jets
                vector<string> frames = {"Higgs", "Top", "W", "Z"};
                if(storeDaughters == true){
                    storeJetDaughters(daughtersOfJet, ijet, boostedDaughters, restJets, frames, jetVecVars, jetColl_ );
                }

                // Fill the jet entry tree
                jetTree->Fill();

            }
          }

          //-------------------------------------------------------------------------------
          // Clear and Reset all tree variables -------------------------------------------
          //-------------------------------------------------------------------------------
          for (unsigned i = 0; i < listOfVars.size(); i++){
            treeVars[ listOfVars[i] ] = -999.99;
          }
          for (unsigned i = 0; i < listOfVecVars.size(); i++){
            jetVecVars[ listOfVecVars[i] ].clear();
          }
          for (unsigned i = 0; i < listOfImgVars.size(); i++){
            // not sure how to reset the image variables or even if we will need to
            //imgVars[ listOfImgVars[i] ]
          }
      }
}


//=================================================================================
// Method called once each job just before starting event loop  -------------------
//=================================================================================

void
BESTProducer::beginStream(edm::StreamID)
{
}

//=================================================================================
// Method called once each job just after ending the event loop  ------------------
//=================================================================================

void
BESTProducer::endStream()
{
}

//=================================================================================
// Method fills 'descriptions' with the allowed parameters for the module  --------
//=================================================================================

void
BESTProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BESTProducer);
