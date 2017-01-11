#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

#include "MyLoss.h"

class ModelParams{
public:
	Alphabet wordAlpha;
	LookupTable words;

	vector<UniParams> projects_params;

public:
	MyLoss loss;

	bool initial(HyperParams& opts, AlignedMemoryPool* mem = NULL){
		// some model parameters should be initialized outside
		if (words.nVSize <= 0){
			cout << "words size is error" << endl;
			return false;
		}
		projects_params.resize(words.nVSize);
		for (int idx = 0; idx < words.nVSize; idx++) {
			projects_params[idx].initial(1, opts.wordDim, false, mem);
		}
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		for (int idx = 0; idx < words.nVSize; idx++) {
			projects_params[idx].exportAdaParams(ada);
		}
	}
	

	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words.E");
	}


	void saveModel(std::ofstream &os) {
		//words.saveEmb(os);
	}

	void loadModel(const string& inFile){

	}
};

#endif 