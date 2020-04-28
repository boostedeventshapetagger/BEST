//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// BESTtoolbox.cpp -------------------------------------------------------------------
//========================================================================================
// C++ file containing functions for use with CMS EDAnalyzer and EDProducer --------------
//////////////////////////////////////////////////////////////////////////////////////////

#include "BESTtoolbox.h"

//========================================================================================
// Calculate Legendre Polynomials --------------------------------------------------------
//----------------------------------------------------------------------------------------
// Simple Legendre polynomial function that can calculate up to order 4 ------------------
// Inputs: argument of the polynomial and order desired ----------------------------------
//----------------------------------------------------------------------------------------

float LegendreP(float x, int order){
   if (order == 0) return 1;
   else if (order == 1) return x;
   else if (order == 2) return 0.5*(3*x*x - 1);
   else if (order == 3) return 0.5*(5*x*x*x - 3*x);
   else if (order == 4) return 0.125*(35*x*x*x*x - 30*x*x + 3);
   else return 0;
}

//========================================================================================
// Calculate Fox Wolfram Moments ---------------------------------------------------------
//----------------------------------------------------------------------------------------
// This function calculates the Fox Wolfram moments for jet constituents -----------------
// in various rest frames. ---------------------------------------------------------------
// Inputs: particles (jet constiuents boosted to rest frame) and empty array that --------
//         that will store the FW moments ------------------------------------------------
//----------------------------------------------------------------------------------------

int FWMoments(std::vector<TLorentzVector> particles, double (&outputs)[5] ){

   // get number of particles to loop over
   int numParticles = particles.size();

   // get energy normalization for the FW moments
   float s = 0.0;
   for(int i = 0; i < numParticles; i++){
   	s += particles[i].E();
   }

   float H0 = 0.0;
   float H4 = 0.0;
   float H3 = 0.0;
   float H2 = 0.0;
   float H1 = 0.0;

   for (int i = 0; i < numParticles; i++){

   	for (int j = i; j < numParticles; j++){

                // calculate cos of jet constituent angles
   		float costh = ( particles[i].Px() * particles[j].Px() + particles[i].Py() * particles[j].Py()
                                   + particles[i].Pz() * particles[j].Pz() ) / ( particles[i].P() * particles[j].P() );
   		float w1 = particles[i].P();
   		float w2 = particles[j].P();

                // calculate legendre polynomials of jet constiteuent angles
   		float fw0 = LegendreP(costh, 0);
   		float fw1 = LegendreP(costh, 1);
   		float fw2 = LegendreP(costh, 2);
   		float fw3 = LegendreP(costh, 3);
   		float fw4 = LegendreP(costh, 4);

                // calculate the Fox Wolfram moments
   		H0 += w1 * w2 * fw0;
   		H1 += w1 * w2 * fw1;
   		H2 += w1 * w2 * fw2;
   		H3 += w1 * w2 * fw3;
   		H4 += w1 * w2 * fw4;

   	}
   }

   // Normalize the Fox Wolfram moments
   if (H0 == 0) H0 += 0.001;      // to prevent dividing by zero
   outputs[0] = (H0);
   outputs[1] = (H1 / H0);
   outputs[2] = (H2 / H0);
   outputs[3] = (H3 / H0);
   outputs[4] = (H4 / H0);

   return 0;
}

//========================================================================================
// Get All Jet Constituents --------------------------------------------------------------
//----------------------------------------------------------------------------------------
// This gets all the jet constituents (daughters) and stores them as a standard ----------
// vector --------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------

void getJetDaughters(std::vector<reco::Candidate * > &daughtersOfJet, std::vector<pat::Jet>::const_iterator jet){
    // First get all daughters for the first Soft Drop Subjet
    for (unsigned int i = 0; i < jet->daughter(0)->numberOfDaughters(); i++){
        if (jet->daughter(0)->daughter(i)->pt() < 0.5) continue;
        daughtersOfJet.push_back( (reco::Candidate *) jet->daughter(0)->daughter(i) );
    }
    // Get all daughters for the second Soft Drop Subjet
    for (unsigned int i = 0; i < jet->daughter(1)->numberOfDaughters(); i++){
        if (jet->daughter(1)->daughter(i)->pt() < 0.5) continue;
        daughtersOfJet.push_back( (reco::Candidate *) jet->daughter(1)->daughter(i));
    }
    // Get all daughters not included in Soft Drop
    for (unsigned int i = 2; i< jet->numberOfDaughters(); i++){
        if (jet->daughter(i)->pt() < 0.5) continue;
        daughtersOfJet.push_back( (reco::Candidate *) jet->daughter(i) );
    }

}

//========================================================================================
// Store Jet Variables -------------------------------------------------------------------
//----------------------------------------------------------------------------------------
// This takes various jet quantaties and stores them on the map used to fill -------------
// the jet tree --------------------------------------------------------------------------
//----------------------------------------------------------------------------------------

void storeJetVariables(std::map<std::string, float> &besVars, std::vector<pat::Jet>::const_iterator jet,
                       int jetColl){
                // pasing a variable with & is pass-by-reference which keeps changes in this func

    // Jet four vector and Soft Drop info
    besVars["jetAK8_phi"] = jet->phi();
    besVars["jetAK8_eta"] = jet->eta();
    besVars["jetAK8_pt"] = jet->pt();
    besVars["jetAK8_mass"] = jet->mass();
    besVars["bDisc"] = jet->bDiscriminator("pfDeepCSVJetTags:probb") + jet->bDiscriminator("pfDeepCSVJetTags:probbb");

    // Store Subjettiness info
    if(jetColl == 0){ // CHS jets
        besVars["jetAK8_Tau4"] = jet->userFloat("NjettinessAK8CHS:tau4");  //important for H->WW jets
        besVars["jetAK8_Tau3"] = jet->userFloat("NjettinessAK8CHS:tau3");
        besVars["jetAK8_Tau2"] = jet->userFloat("NjettinessAK8CHS:tau2");
        besVars["jetAK8_Tau1"] = jet->userFloat("NjettinessAK8CHS:tau1");
	besVars["jetAK8_Tau21"] = jet->userFloat("NjettinessAK8CHS:tau2") / jet->userFloat("NjettinessAK8CHS:tau1");
        besVars["jetAK8_Tau32"] = jet->userFloat("NjettinessAK8CHS:tau3") / jet->userFloat("NjettinessAK8CHS:tau2");
	besVars["jetAK8_SoftDropMass"] = jet->userFloat("ak8PFJetsCHSValueMap:ak8PFJetsCHSSoftDropMass");
    }
    if(jetColl == 1){ // PUPPI jets
        besVars["jetAK8_Tau4"] = jet->userFloat("NjettinessAK8Puppi:tau4");  //important for H->WW jets
        besVars["jetAK8_Tau3"] = jet->userFloat("NjettinessAK8Puppi:tau3");
        besVars["jetAK8_Tau2"] = jet->userFloat("NjettinessAK8Puppi:tau2");
        besVars["jetAK8_Tau1"] = jet->userFloat("NjettinessAK8Puppi:tau1");
	besVars["jetAK8_Tau21"] = jet->userFloat("NjettinessAK8Puppi:tau2") / jet->userFloat("NjettinessAK8Puppi:tau1");
        besVars["jetAK8_Tau32"] = jet->userFloat("NjettinessAK8Puppi:tau3") / jet->userFloat("NjettinessAK8Puppi:tau2");
	besVars["jetAK8_SoftDropMass"] = jet->userFloat("ak8PFJetsPuppiSoftDropMass");
	auto subjets = jet->subjets("SoftDropPuppi");
	if (subjets.size() < 2){
	  std::cout << "This will exit, not enough subjets" << std::endl;
	  exit(1);
	}
	if (!subjets[0]){
	  std::cout << "This will exit, invalid subjet 0" << std::endl;
          exit(1);
        }
        if (!subjets[1]){
	  std::cout << "This will exit, invalid subjet 1" << std::endl;
          exit(1);
	}
	besVars["bDisc1"] = subjets[0]->bDiscriminator("pfDeepCSVJetTags:probb") + subjets[0]->bDiscriminator("pfDeepCSVJetTags:probbb");
        besVars["bDisc2"] = subjets[1]->bDiscriminator("pfDeepCSVJetTags:probb") + subjets[1]->bDiscriminator("pfDeepCSVJetTags:probbb");

    }
}

//========================================================================================
// Store Secondary Vertex Information ----------------------------------------------------
//----------------------------------------------------------------------------------------
// This takes various secondary vertex quantities and stores them on the map -------------
// used to fill the tree -----------------------------------------------------------------
//----------------------------------------------------------------------------------------

void storeSecVertexVariables(std::map<std::string, float> &besVars,
                             std::map<std::string, std::vector<float> > &jetVecVars, TLorentzVector jet,
                             std::vector<reco::VertexCompositePtrCandidate> secVertices){

   int numMatched = 0; // counts number of secondary vertices
   for(std::vector<reco::VertexCompositePtrCandidate>::const_iterator vertBegin = secVertices.begin(),
              vertEnd = secVertices.end(), ivert = vertBegin; ivert != vertEnd; ivert++){
      TLorentzVector vert(ivert->px(), ivert->py(), ivert->pz(), ivert->energy() );
      // match vertices to jet
      if(jet.DeltaR(vert) < 0.8 ){
         numMatched++;
         // save secondary vertex info for the first three sec vertices
         jetVecVars["SV_pt"].push_back(ivert->pt() );
         jetVecVars["SV_eta"].push_back(ivert->eta() );
         jetVecVars["SV_phi"].push_back(ivert->phi() );
         jetVecVars["SV_mass"].push_back(ivert->mass() );
         jetVecVars["SV_nTracks"].push_back(ivert->numberOfDaughters() );
         jetVecVars["SV_chi2"].push_back(ivert->vertexChi2() );
         jetVecVars["SV_Ndof"].push_back(ivert->vertexNdof() );
      }
   }
   besVars["nSecondaryVertices"] = numMatched;
}

//========================================================================================
// Calculate BEST Rest Frame Variables ---------------------------------------------------
//----------------------------------------------------------------------------------------
// This boosts an ak8 jet (and all of its constituents) into heavy object rest frame -----
// and then uses it to calculate FoxWolfram moments, Event Shape Variables, --------------
// and assymmetry variables --------------------------------------------------------------
//----------------------------------------------------------------------------------------

void calcBESvariables(std::map<std::string, float> &besVars, std::vector<reco::Candidate *> &daughtersOfJet,
                      std::map<std::string, std::vector<TLorentzVector> > &boostedDaughters,
                      std::vector<pat::Jet>::const_iterator jet, std::map<std::string, std::vector<fastjet::PseudoJet> > &restJets,
                      std::map<std::string, std::array<std::array<std::array<float, 1>, 31>, 31> > &imgVars,
                      std::string frame, float mass){

    // get 4 vector for heavy object rest frame
    typedef reco::Candidate::PolarLorentzVector fourv;
    fourv thisJet = jet->polarP4();
    TLorentzVector thisJetLV(0.,0.,0.,0.);
    thisJetLV.SetPtEtaPhiM(thisJet.Pt(), thisJet.Eta(), thisJet.Phi(), mass );

    std::vector<TLorentzVector> particles;
    std::vector<math::XYZVector> particles2;
    std::vector<reco::LeafCandidate> particles3;
    std::vector<fastjet::PseudoJet> FJparticles;
    std::vector<TLorentzVector> boostedCands; // Possible bug! = new std::vector<TLorentzVector>;

    // 4 vectors to be filled with reclustered jet additions
    TLorentzVector jet12LV(0.,0.,0.,0.);
    TLorentzVector jet13LV(0.,0.,0.,0.);
    TLorentzVector jet23LV(0.,0.,0.,0.);
    TLorentzVector jet1234LV(0.,0.,0.,0.);

    double sumPz = 0;
    double sumP = 0;

    // Boost to object rest frame
    for(unsigned int i = 0; i < daughtersOfJet.size(); i++){
        // Do not include low pT particles
        if (daughtersOfJet[i]->pt() < 0.5) continue;

        // Create 4 vector to boost to Higgs frame
        TLorentzVector thisParticleLV( daughtersOfJet[i]->px(), daughtersOfJet[i]->py(), daughtersOfJet[i]->pz(), daughtersOfJet[i]->energy() );

        // Boost to heavy object rest frame
        thisParticleLV.Boost( -thisJetLV.BoostVector() );

        // Store candidate information for making the images
        boostedCands.push_back(thisParticleLV);

        // Now that PF candidates are stored, make the boost axis the Z-axis
        // Important for BES variables
        pboost( thisJetLV.Vect(), thisParticleLV.Vect(), thisParticleLV);

        particles.push_back( thisParticleLV );
        particles2.push_back( math::XYZVector( thisParticleLV.X(), thisParticleLV.Y(), thisParticleLV.Z() ));
        particles3.push_back( reco::LeafCandidate(+1, reco::Candidate::LorentzVector( thisParticleLV.X(), thisParticleLV.Y(),
                                                                                        thisParticleLV.Z(), thisParticleLV.T() ) ));
        FJparticles.push_back( fastjet::PseudoJet( thisParticleLV.X(), thisParticleLV.Y(), thisParticleLV.Z(), thisParticleLV.T() ) );

        // Sum rest frame momenta for asymmetry calculation, but only for pt > 10
	//Why?????
	//        if (daughtersOfJet[i]->pt() < 10) continue;
        sumPz += thisParticleLV.Pz();
        sumP += abs( thisParticleLV.P() );
    }

    // make the rest frame jet images
    boostedDaughters[frame+"Frame"] = boostedCands;
    imgVars[frame+"Frame_image"] = boostedJetCamera(boostedCands);

    // Fox Wolfram Moments
    double fwm[5] = { 0.0, 0.0 ,0.0 ,0.0,0.0};
    FWMoments( particles, fwm);
    besVars["FoxWolfH1_"+frame] = fwm[1];
    besVars["FoxWolfH2_"+frame] = fwm[2];
    besVars["FoxWolfH3_"+frame] = fwm[3];
    besVars["FoxWolfH4_"+frame] = fwm[4];

    // Event Shape Variables
    EventShapeVariables eventShapes( particles2 );
    Thrust thrustCalculator( particles3.begin(), particles3.end() );
    besVars["isotropy_"+frame]   = eventShapes.isotropy();
    besVars["sphericity_"+frame] = eventShapes.sphericity();
    besVars["aplanarity_"+frame] = eventShapes.aplanarity();
    besVars["thrust_"+frame]     = thrustCalculator.thrust();

    // Jet Asymmetry
    double asymmetry             = sumPz/sumP;
    besVars["asymmetry_"+frame] = asymmetry;

    // Recluster the jets in the heavy object rest frame
    fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, 0.4);
    fastjet::ClusterSequence cs(FJparticles, jet_def);
    //    std::vector<fastjet::PseudoJet> jetsFJ = sorted_by_pt(cs.inclusive_jets(20.0));
    //Changed to 0.0 here, the 20.0 cuts on pT relative to a meaningless axis
    //    std::vector<fastjet::PseudoJet> jetsFJ = sorted_by_pt(cs.inclusive_jets(0.0));
    std::vector<fastjet::PseudoJet> jetsFJ = sorted_by_E(cs.inclusive_jets(0.0));
    restJets[frame+"Frame"] = jetsFJ;

    // Store reclustered jet info
    for(unsigned int i = 0; i < jetsFJ.size(); i++){

        // make a TLorentzVector for the current clustered rest frame jet
        TLorentzVector iJetLV(jetsFJ[i].px(), jetsFJ[i].py(), jetsFJ[i].pz(), jetsFJ[i].e() );

        // get fest frame jet four vector combinations
        switch(i){
        case 0:
	  //            jet12LV   = jet12LV   + iJetLV;
	  //            jet13LV   = jet13LV   + iJetLV;
	  //            jet1234LV = jet1234LV + iJetLV;
            jet12LV   = iJetLV;
            jet13LV   = iJetLV;
            jet1234LV = iJetLV;

            break;
        case 1:
	  besVars["jet12_DeltaCosTheta_"+frame]   = (jet12LV.Vect()).Dot(iJetLV.Vect()) / (jet12LV.Vect().Mag() * iJetLV.Vect().Mag());
            jet12LV   = jet12LV   + iJetLV;
	    //            jet23LV   = jet23LV   + iJetLV;
            jet23LV   = iJetLV;
            jet1234LV = jet1234LV + iJetLV;
            break;
        case 2:
	  besVars["jet13_DeltaCosTheta_"+frame]   = (jet13LV.Vect()).Dot(iJetLV.Vect()) / (jet13LV.Vect().Mag() * iJetLV.Vect().Mag());
            jet13LV   = jet13LV   + iJetLV;
	    besVars["jet23_DeltaCosTheta_"+frame]   = (jet23LV.Vect()).Dot(iJetLV.Vect()) / (jet23LV.Vect().Mag() * iJetLV.Vect().Mag());
            jet23LV   = jet23LV   + iJetLV;
            jet1234LV = jet1234LV + iJetLV;
            break;
        case 3:
            jet1234LV = jet1234LV + iJetLV;
            break;
        }
    }

    // Store reclustered jet mass combinations
    besVars["jet12_mass_"+frame]   = jet12LV.M();
    besVars["jet13_mass_"+frame]   = jet13LV.M();
    besVars["jet23_mass_"+frame]   = jet23LV.M();
    besVars["jet1234_mass_"+frame] = jet1234LV.M();
    besVars["jet12_CosTheta_"+frame]   = jet12LV.CosTheta();
    besVars["jet13_CosTheta_"+frame]   = jet13LV.CosTheta();
    besVars["jet23_CosTheta_"+frame]   = jet23LV.CosTheta();
    besVars["jet1234_CosTheta_"+frame] = jet1234LV.CosTheta();
    besVars["nJets_"+frame] = jetsFJ.size();
}

//========================================================================================
// Store Jet Constituents ----------------------------------------------------------------
//----------------------------------------------------------------------------------------
// This stores all the jet constituents in a vector corresponding to a jetTree so these --
//  variables can be used in training BEST -----------------------------------------------
//----------------------------------------------------------------------------------------

void storeJetDaughters(std::vector<reco::Candidate * > &daughtersOfJet, std::vector<pat::Jet>::const_iterator jet,
                       std::map<std::string, std::vector<TLorentzVector> > &boostedDaughters,
                       std::map<std::string, std::vector<fastjet::PseudoJet> > &restJets, std::vector<std::string> frames,
                       std::map<std::string, std::vector<float> > &jetVecVars, int jetColl ){

    // loop over lab frame candidates
    for(unsigned int i = 0; i < daughtersOfJet.size(); i++){

        // Do not include low pT particles
        if (daughtersOfJet[i]->pt() < 0.5) continue;

        // Store the candidate
        jetVecVars["LabFrame_PF_candidate_px"].push_back(daughtersOfJet[i]->px() );
        jetVecVars["LabFrame_PF_candidate_py"].push_back(daughtersOfJet[i]->py() );
        jetVecVars["LabFrame_PF_candidate_pz"].push_back(daughtersOfJet[i]->pz() );
        jetVecVars["LabFrame_PF_candidate_energy"].push_back(daughtersOfJet[i]->energy() );

        // PUPPI weights for puppi jets
        if (jetColl == 1){
            pat::PackedCandidate *iparticle = (pat::PackedCandidate *) daughtersOfJet[i];
	    if(!iparticle){
	      std::cout<<"ERROR: The PF candidate did not get properly converted to PackedCandidate"<<std::endl;
	      std::cout<<" 'Transfiguration is some of the most dangerous and complex magic!'"<<std::endl;
	      exit(1);
	    }
            jetVecVars["PUPPI_weights"].push_back( iparticle->puppiWeight() );
        }
    }


    // loop over rest frames
    for(unsigned int iFrame = 0; iFrame < frames.size(); iFrame++){

        std::string frame = frames[iFrame];

        // loop over candidates in the rest frame
        for(auto icand = boostedDaughters[frame+"Frame"].begin(); icand != boostedDaughters[frame+"Frame"].end(); icand++){

            // store the rest frame candidate
            jetVecVars[frame+"Frame_PF_candidate_px"].push_back(icand->Px() );
            jetVecVars[frame+"Frame_PF_candidate_py"].push_back(icand->Py() );
            jetVecVars[frame+"Frame_PF_candidate_pz"].push_back(icand->Pz() );
            jetVecVars[frame+"Frame_PF_candidate_energy"].push_back(icand->E() );

        }

        // loop over rest frame jets
        for(auto ijet = restJets[frame+"Frame"].begin(); ijet != restJets[frame+"Frame"].end(); ijet++){

            // store the rest frame jet information
            jetVecVars[frame+"Frame_jet_px"].push_back(ijet->px());
            jetVecVars[frame+"Frame_jet_py"].push_back(ijet->py());
            jetVecVars[frame+"Frame_jet_pz"].push_back(ijet->pz());
            jetVecVars[frame+"Frame_jet_energy"].push_back(ijet->e());
        }

    }

}

//========================================================================================
// Boosted Jet Camera --------------------------------------------------------------------
//----------------------------------------------------------------------------------------
// Make the rest frame jet images that will be run through BEST --------------------------
// boostedCands = the lorentz vectores of the the jet PF candidates in the rest frame-
// Image = the container for the jet image -----------------------------------------------
//----------------------------------------------------------------------------------------

std::array<std::array<std::array<float, 1>, 31>, 31> boostedJetCamera(std::vector<TLorentzVector> &boostedCands){

    // create a place to store the image
    std::array<std::array<std::array<float, 1>, 31>, 31> Image;

    //Sort the new list of particle flow candidates in the rest rame by energy
    auto sortLambda = [] (const TLorentzVector& lv1, const TLorentzVector& lv2) {return lv1.E() > lv2.E(); };
    std::sort(boostedCands.begin(), boostedCands.end(), sortLambda);

    //------------------------------------------------------------------------------------
    // Rotations in the rest frame -------------------------------------------------------
    //------------------------------------------------------------------------------------

    // define the rotation angles for the first two rotations
    float rotPhi = boostedCands.at(0).Phi();
    float rotTheta = boostedCands.at(0).Theta();

    // perform the first two rotations and sum energy
    float sumE = 0;
    float candNum = 0;
    for(auto icand = boostedCands.begin(); icand != boostedCands.end(); icand++){

        // sum energy
        sumE+= icand->E();

        // rotate all candidates so that the leading candidate is in the x y plane
        icand->RotateZ(-rotPhi);

        // make sure the leading candidate has been fully rotated
        if(candNum == 0) icand->SetPy(0);

        // rotate all candidates so that the leading candidate is on the x axis
        icand->RotateY(TMath::Pi()/2.0 - rotTheta);

        // make sure the leading candidate has been fully rotated
        if(candNum == 0) icand->SetPz(0);

        candNum++;
    }

    // create the rotation angle for the third rotation
    float subPsi = TMath::ATan2(boostedCands.at(1).Py(), boostedCands.at(1).Pz());

    candNum = 0;
    for(auto icand = boostedCands.begin(); icand != boostedCands.end(); icand++){

        // rotate all candidates about the x axis so that the subleading candidate is in the xy plane
        icand->RotateX(subPsi - TMath::Pi()/2.0);

        // make sure the leading candidate has been fully rotated
        if(candNum == 1) icand->SetPz(0);

        candNum++;
    }

    //Reflect if necessary
    float rightSum = 0;
    float leftSum = 0;
    float topSum = 0;
    float botSum = 0;
    for(auto icand = boostedCands.begin(); icand != boostedCands.end(); icand++){
        if (icand->CosTheta() > 0){
            topSum+=icand->E();
        }
        else if (icand->CosTheta() < 0){
            botSum+=icand->E();
        }

        if (icand->Phi() > 0){
            rightSum+=icand->E();
        }
        else if (icand->Phi() < 0){
            leftSum+=icand->E();
        }
    }
    //Initialize image with 0's in all bins
    for (int nx = 0; nx < 31; nx++){
        for (int ny= 0; ny < 31; ny++){
            Image[nx][ny][0] = 0;
        }
    }
    //find the x and y coordinates in phi, theta binned space
    //Then fill Image with normalized energy

    for(auto icand = boostedCands.begin(); icand != boostedCands.end(); icand++){
        int x_bin = -1;
        int y_bin = -1;
        if (topSum >= botSum){
            y_bin = static_cast<int>(31*(icand->CosTheta() + 1)/(2.0));
            y_bin = y_bin%31;
        }
        else{
            y_bin = static_cast<int>(31*(-icand->CosTheta() + 1)/(2.0));
            y_bin = y_bin%31;
        }
        if (rightSum >= leftSum){
            x_bin = static_cast<int>(31*(icand->Phi() + TMath::Pi())/(2.0 * TMath::Pi()));
            x_bin = x_bin%31;
        }
        else{
            x_bin = static_cast<int>(31*(-icand->Phi() + TMath::Pi())/(2.0 * TMath::Pi()));
            x_bin = x_bin%31;
        }
        Image[x_bin][y_bin][0] += icand->E()/sumE * 10 ;
    }
    return Image;
}


//========================================================================================
// Make boost axis the rest frame z axis -------------------------------------------------
//----------------------------------------------------------------------------------------
// Given jet constituent lab momentum, find momentum relative to beam direction pbeam ----
// plab = Particle 3-vector in Boost Frame -----------------------------------------------
// pbeam = Lab Jet 3-vector --------------------------------------------------------------
//----------------------------------------------------------------------------------------

void pboost( TVector3 pbeam, TVector3 plab, TLorentzVector &pboo ){

    double pl = plab.Dot(pbeam);
    pl *= double(1. / pbeam.Mag());

    // set x axis direction along pbeam x (0,0,1)
    TVector3 pbx;

    pbx.SetX(pbeam.Y());
    pbx.SetY(-pbeam.X());
    pbx.SetZ(0.0);

    pbx *= double(1. / pbx.Mag());

    // set y axis direction along -pbx x pbeam
    TVector3 pby;

    pby = -pbx.Cross(pbeam);
    pby *= double(1. / pby.Mag());

    pboo.SetX((plab.Dot(pbx)));
    pboo.SetY((plab.Dot(pby)));
    pboo.SetZ(pl);

}
