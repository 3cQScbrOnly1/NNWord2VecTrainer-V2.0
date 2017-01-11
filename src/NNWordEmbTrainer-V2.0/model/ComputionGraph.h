#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	// node instances
	vector<LookupNode> _contexts;

	PAddNode _x;
	vector<UniNode> _projects;
	PNode _Poutput;

	
public:
	ComputionGraph() : Graph(){

	}

	~ComputionGraph(){
		clear();
	}

	inline void createNode(int contextSize, int wordSize){
		_contexts.resize(contextSize);
		_projects.resize(wordSize);
	}

public:
	inline void clear(){
		Graph::clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){
		int context_size = _contexts.size();
		for (int idx = 0; idx < context_size; idx++) {
			_contexts[idx].setParam(&model.words);
			_contexts[idx].init(opts.wordDim, -1, mem);
		}
		_x.init(opts.wordDim, -1, mem);
		int output_size = _projects.size();
		for (int idx = 0; idx < output_size; idx++) {
			_projects[idx].setParam(&model.projects_params[idx]);
			_projects[idx].init(1, -1, mem);
			_projects[idx].setFunctions(&fsigmoid, &dsigmoid);
		}
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Feature& feature, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation
		// second step: build graph
		//forward
		int context_size = feature.context_word_indexs.size();
		for (int idx = 0; idx < context_size; idx++){
			_contexts[idx].forward(this, feature.context_word_indexs[idx]);
		}

		_x.forward(this, getPNodes(_contexts, context_size));
		int pos_index = feature.target_word_index;
		_projects[pos_index].forward(this, &_x);
		_Poutput = &_projects[pos_index];
	}
};

#endif /* SRC_ComputionGraph_H_ */