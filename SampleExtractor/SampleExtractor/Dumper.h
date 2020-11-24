#pragma once

#include "def.h"

using namespace std;

static string startvm = " && VBoxManage startvm ";
static string dbgvm = " && VBoxManage debugvm ";
static string dumpcore = " dumpvmcore --filename=";

class Dumper {
private:
	string VMname;
	string VMwarePath;
public:
	Dumper(string name = "", string path = "") :VMname(name), VMwarePath(path){};
	void start_vm()const;
	string dump_mem(string)const;
};

void Dumper::start_vm() const{
	string cmd(VMwarePath + startvm + VMname);
	system(cmd.c_str());
}

// VBoxManage debugvm <uuid|vmname> dumpvmcore [--filename=name]
string Dumper::dump_mem(string input)const {
	string filename;
	if(input.empty())
		filename = DATAFILEPATH + VMname + ".efl";
	else
		filename = DATAFILEPATH + VMname + "_" + input + ".efl";
	string cmd = VMwarePath + dbgvm + VMname + dumpcore + filename;
	cout << cmd << endl;
	system(cmd.c_str());
	return filename;
}