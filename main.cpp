#include "ID3.h"
#include "C4d5.h"
#include <iostream>

using std::cout;

int main(int argc, char* argv[])
{
    map <string, string> example;
    example.insert(map<string, string>::value_type("templature", "89"));
    example.insert(map<string, string>::value_type("outlook", "sunny"));
	
    example.insert(map<string, string>::value_type("humidity", "90"));
    C4d5 *test = new C4d5("test.txt");
	test->BuildDecisionTree(0.1);
	test->PessimisticPruning();
	test->PrintDecisionTree();
    cout<<test->Predict(example);
	test->~C4d5();
	return 0;
}



