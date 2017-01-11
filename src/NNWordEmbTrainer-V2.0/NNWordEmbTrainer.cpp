#include "NNWordEmbTrainer.h"

#include "Argument_helper.h"

Trainer::Trainer(int memsize, int graphnum, int buffersize):m_driver(memsize, graphnum), buffer_size(buffersize){
	instances_count = 0;
	context_size = 2;
	table_size = 1e4;
	neg_word_size = 5;
	table.resize(table_size);
	START = "-start-";
	END = "-end-";
}

Trainer::~Trainer(){
}

void Trainer::createWordStates(const string& file_name) {
	m_pipe.initInputFile(file_name.c_str());
	Instance *pInstance = m_pipe.nextInstance();
	vector<Instance> insts;
	insts.clear();
	while (pInstance) {
		Instance trainInstance;
		trainInstance.copyValuesFrom(*pInstance);
		insts.emplace_back(trainInstance);
		if (insts.size() == buffer_size) {
			addWord2Stats(insts);
			insts.clear();
		}
		pInstance = m_pipe.nextInstance();
	}
	if (insts.size() != 0)
		addWord2Stats(insts);
	m_pipe.uninitInputFile();
	m_word_stats[START] = m_options.wordCutOff + 1;
	m_word_stats[END] = m_options.wordCutOff + 1;
	cout << endl << "word size: "<< m_word_stats.size() << endl;
}


void Trainer::createRandomTable(){
	int vocab_size = m_word_stats.size();
	if (vocab_size == 0) {
		cout << "error word count" << endl;
		return;
	}
	if (m_driver._modelparams.wordAlpha.size() == 0) {
		cout << "error wordAlpha in modelparam " << endl;
		return;
	}
	// copy m_word_stats;
	for (unordered_map<string, int>::iterator it = m_word_stats.begin();
		it != m_word_stats.end(); it++) {
		if (it->second > m_options.wordCutOff) {
			vocab.emplace_back(make_pair(it->first, it->second));
		}
	}

	double d1, power = 0.75, train_words_pow = 0;
	for (int idx = 0; idx < vocab_size; idx++) {
		train_words_pow += pow(vocab[idx].second , power);
	}
	int i = 0;
	d1 = pow(vocab[i].second, power)/ train_words_pow;
	for (int a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (double)table_size > d1) {
			i++;
			d1 += pow(vocab[i].second, power) / train_words_pow;
		}
		if (i >= vocab_size)
			i = vocab_size - 1;
	}
}

void Trainer::addPosExample(const int& target_word_index, const vector<int>& context_word_indexs, vector<Example>& vecExam){
	Example exam;
	exam.m_feature.target_word_index = target_word_index;
	exam.m_feature.context_word_indexs = context_word_indexs;
	exam.pos_label();
	vecExam.push_back(exam);
}

void Trainer::addNegExample(const int& target_word_index, const vector<int>& context_word_indexs, vector<Example>& vecExam){
	int neg_word_index;
	bool IS_NEG = true;
	Example exam;
	for (int i = 0; i < neg_word_size; i++) {
		exam.clear();
		int table_index = rand() % table_size;
		neg_word_index = table[table_index];
		if (neg_word_index == target_word_index) {
			IS_NEG = false;
			break;
		}
		if (IS_NEG) {
			exam.m_feature.target_word_index = neg_word_index;
			exam.m_feature.context_word_indexs = context_word_indexs;
			exam.neg_label();
			vecExam.push_back(exam);
		}
	}
}

void Trainer::addWord2Stats(const vector<Instance>& vecInsts){
	int numInstance = vecInsts.size();
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<string> &words = pInstance->m_words;
		int words_num = words.size();
		for (int i = 0; i < words_num; i++)
		{
			//string curword = normalize_to_lowerwithdigit(words[i]);
			m_word_stats[words[i]]++;
		}
		if (m_options.maxInstance > 0 && instances_count == m_options.maxInstance)
			break;
	}
	instances_count += numInstance;
	cout << instances_count << " ";
}

void Trainer:: convert2Examples(const Instance* pInstance, vector<Example>& vecExam){
	const vector<string>& words = pInstance->m_words;
	vector<string> neg_words;
	int word_size = words.size();
	string curr_context_word;
	int target_word_index, context_word_index;
	vector<int> context_word_indexs;
	for (int idx = 0; idx < word_size; idx++) {
		target_word_index = m_driver._modelparams.wordAlpha.from_string(words[idx]);
		context_word_indexs.clear();
		for (int offset = 1; offset <= context_size; offset++) {
			neg_words.clear();
			if (idx - offset < 0)
				curr_context_word = START;
			else
				curr_context_word = words[idx - offset];

			context_word_index = m_driver._modelparams.wordAlpha.from_string(curr_context_word);
			context_word_indexs.emplace_back(context_word_index);
		}

		for (int offset = 1; offset <= context_size; offset++) {
			neg_words.clear();
			if (idx + offset >= word_size)
				curr_context_word = END;
			else
				curr_context_word = words[idx + offset];

			context_word_index = m_driver._modelparams.wordAlpha.from_string(curr_context_word);
			context_word_indexs.emplace_back(context_word_index);
		}
		addPosExample(target_word_index, context_word_indexs, vecExam);
		addNegExample(target_word_index, context_word_indexs, vecExam);
	}
}

void Trainer::train(const string& trainFile, const string& modelFile, const string& optionFile) {
	if (optionFile != "")
		m_options.load(optionFile);
	m_driver._hyperparams.setRequared(m_options);
	cout << "Create Alphabet....." << endl;
	createWordStates(trainFile);
	m_driver._modelparams.wordAlpha.initial(m_word_stats, m_options.wordCutOff);
	m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha, m_options.wordEmbSize, true);
	m_driver._modelparams.wordAlpha.set_fixed_flag(true);
	m_driver.initial();
	createRandomTable();
	time_t start_time = time(NULL);
	trainEmb(trainFile);
	time_t end_time = time(NULL);
	cout << endl << "Training cost time :" << end_time - start_time  << endl;
	cout << "Saving model..." << endl;
	writeModelFile(modelFile);
	cout << "Save complete!" << endl;
}

void Trainer::writeModelFile(const string& outputModelFile) {
	ofstream os(outputModelFile);
	if (os.is_open())
	{
		m_driver._modelparams.saveModel(os);
	}
	else
	{
		cout << "write model error." << endl;
	}
}

dtype Trainer::trainInstances(const vector<Instance>& vecInst){
	int vecSize = vecInst.size();
	int examSize;
	dtype cost = 0;
	vector<vector<Example> > all_exams(vecSize);
	time_t start_time = time(NULL);
#pragma omp parallel for
	for (int idx = 0; idx < vecSize; idx++) {
		convert2Examples(&vecInst[idx], all_exams[idx]);
	}
	time_t end_time = time(NULL);
	cout << "Gather examples cost time :" << end_time - start_time << endl;

	start_time = time(NULL);
	vector<Example> vecExams;
	for (int idx = 0; idx < vecSize; idx++) {
		vecExams.insert(vecExams.end(), all_exams[idx].begin(), all_exams[idx].end());
	}
	end_time = time(NULL);
	cout << "insert time :" << end_time - start_time << endl;

	random_shuffle(vecExams.begin(), vecExams.end());
	cout << "exam size:" << vecExams.size() << endl;
	end_time = time(NULL);

	start_time = time(NULL);
	cost += m_driver.train(vecExams, 1);
	m_driver.updateModel();
	end_time = time(NULL);
	cout << "one buffer cost time :" << end_time - start_time << endl;

	return cost;
}

void Trainer::trainEmb(const string& trainFile){
	m_pipe.initInputFile(trainFile.c_str());
	Instance *pInstance = m_pipe.nextInstance();
	vector<Instance> insts;
	insts.clear();
	dtype cost;
	int count = 0;
	while (pInstance) {
		Instance trainInstance;
		trainInstance.copyValuesFrom(*pInstance);
		insts.emplace_back(trainInstance);
		if (insts.size() == buffer_size) {
			cost = trainInstances(insts);
			cout << "cost: " << cost << endl;
			insts.clear();
			count+=buffer_size;
			cout << "count: " << count << endl;
		}
		pInstance = m_pipe.nextInstance();
	}
	if (insts.size() > 0)
	{
		cost = trainInstances(insts);
		cout << "cost: " << cost << endl;
		count += buffer_size;
		cout << "count: " << count << endl;
	}
	m_pipe.uninitInputFile();
}

int main(int argc, char* argv[]) {

	std::string trainFile = "", devFile = "", testFile = "", modelFile = "", optionFile = "";
	std::string outputFile = "";
	bool bTrain = false;
	int memsize = 0;
	int graphNum = 100;
	int threadNum = 1;
	int bufferSize = 100;
	dsr::Argument_helper ah;

	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_int("memsize", "memorySize", "named_int", "This argument decides the size of static memory allocation", memsize);
	ah.new_named_int("thread", "thread num", "named_int", "The thread size", threadNum);
	ah.new_named_int("buffer", "buffer size", "named_int", "The buffer size", bufferSize);

	ah.process(argc, argv);

	if (memsize < 0)
		memsize = 0;
	//omp_set_num_threads(thread);
	cout << "Thread num: "<<  threadNum << endl;
//	omp_set_num_threads(threadNum);

	Trainer the_trainer(memsize, threadNum, bufferSize);
	the_trainer.train(trainFile, modelFile, optionFile);
	/*
	if (bTrain) {
		the_classifier.train(trainFile, devFile, testFile, modelFile, optionFile);
	}
	else {
		the_classifier.test(testFile, outputFile, modelFile);
	}
	*/
	//test(argv);
	//ah.write_values(std::cout);
}