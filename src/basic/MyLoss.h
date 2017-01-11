#ifndef _MYLOSS_H_
#define _MYLOSS_H

#include "MyLib.h"
#include "Metric.h"
#include "Node.h"


struct MyLoss{
public:
	inline void loss(PNode x, const dtype &answer){
		int nDim = x->dim;
		if (nDim != 1) {
			std::cerr << "MyLoss error: dim size invaild" << std::endl;
			return;
		}
		x->lossed = true;
		x->loss = answer - x->val[0];
	}
};

#endif // _MYLOSS_H_