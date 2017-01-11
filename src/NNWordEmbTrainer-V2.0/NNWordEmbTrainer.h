#include "N3L.h"

#include "Options.h"
#include "Instance.h"
#include "Example.h"
#include "Driver.h"
#include "Pipe.h"
#include "Utf.h"


using namespace nr;
using namespace std;

class Trainer{
public:
	unordered_map<string, int> m_word_stats;
	vector<pair<string, int> > vocab;
	int instances_count;
	int neg_word_size;
	int buffer_size;
	int context_size;
	int error_size;
	vector<int> table;
	int table_size;
	string START;
	string END;
public:
	Options m_options;
	
	Driver m_driver;

	Pipe m_pipe;

public:
	Trainer(int memsize, int threadnum, int buffersize);
	virtual ~Trainer();

public:
	void createWordStates(const string& file_name);
	void addWord2Stats(const vector<Instance>& insts);
	void train(const string& trainFile, const string& modelFile, const string& optionFile);
	void trainEmb(const string& trainFile);
	dtype trainInstances(const vector<Instance>& vecInst);
	void createRandomTable();


	void addPosExample(const int& target_word_index, const vector<int>& context_word_indexs, vector<Example>& vecExam);
	void addNegExample(const int& target_word_index, const vector<int>& context_word_indexs, vector<Example>& vecExam);

	void convert2Examples(const Instance* pInstance, vector<Example>& vecExam);
	void writeModelFile(const string& outputModelFile);
};