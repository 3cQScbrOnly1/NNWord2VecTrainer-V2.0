#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

#include <iostream>
#include <vector>

using namespace std;

class Feature {
public:
	int target_word_index;
	vector<int> context_word_indexs;
public:
	Feature(){}
	void clear() {
		target_word_index = -1;
		context_word_indexs.clear();
	}
};

class Example {
private:
public:
	Feature m_feature;
	double m_label;
public:
	Example(){
	}

	void neg_label(){
		m_label = 1;
	}

	void pos_label(){
		m_label = 0;
	}

	void clear() {
		m_feature.clear();
	}
};

#endif /*_EXAMPLE_H_*/