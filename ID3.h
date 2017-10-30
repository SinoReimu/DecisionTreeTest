/*
 * ID3.h
 *
 *  Created on: Sep 19, 2015
 *      Author: th
 */

#ifndef ID3_H_
#define ID3_H_

#include <string>
#include <map>
#include <vector>

using std::vector;
using std::map;
using std::string;

class ID3{
private:
	bool has_real;
protected:
	int minLeaf;
	string relation;
	string forecasting_attribute_name; //记录需要预测的属性名
	int forecasting_attribute_value_length; //记录需要预测的属性的分支数
	map<string, vector<string> > Attribute_Name_Values; //记录属性名和对应的属性值
	vector<string> Attribute_Names; //记录属性名
	vector<map<string, string> > Samples; //用来训练的实例
	vector<map<string, string> > Test_Samples; //用来测试的实例
	struct DecisionTreeNode{
		string attribute_name;
		string attribute_pre_value;
		string majority_forecasting_attribute_value; //当前剩余实例中最多的forecasting属性值
		int E; int N; //用于悲观剪枝E错误实例数；N当前总实例数
		vector<DecisionTreeNode*> children;
		DecisionTreeNode() :attribute_name(""), attribute_pre_value(""), majority_forecasting_attribute_value(""), E(-1), N(-1){};
	} *root;

	double Information(const map<string, int>& amount, int number)const;
	double ComputeEntropy(const vector<map<string, string> >& remain_Samples, string attribute_name, string attribute_value, int& number)const;
	double ComputeEntropy(const vector<map<string, string> >& remain_Samples, int& number)const;
	double ComputeGain(const vector<map<string, string> >& remain_Samples, string attribute_name)const;
	string IsPure(const vector<map<string, string> >& remain_Samples)const;
	string MajorityDecision(const vector<map<string, string> >& remain_Samples)const;
	virtual DecisionTreeNode* BuildDiscreteDecisionTree(DecisionTreeNode* current_node, vector<map<string, string> > remain_Samples, vector<string> remain_Attribute_Names);
	void PrintDecisionTree(DecisionTreeNode* node, int depth)const;
	void FreeDecisionTree(DecisionTreeNode* node);
	bool Gethasreal()const;
	void Sethasreal(bool flag);
public:
	ID3(string path, int minLeaf = 2);
	void PrintDecisionTree()const;
	virtual void BuildDecisionTree();
	virtual void BuildDecisionTree(double percentage);
	virtual string Predict(map<string,string> example)const;
	double Predict()const;
	virtual void PessimisticPruning(const double z = 0.69){};
	void test();
	virtual ~ID3();
};

#endif
