// BEAM OpenCL Miner
// Main Function
// Copyright 2018 The Beam Team	
// Copyright 2018 Wilke Trei

#include "beamStratum.h"
#include "cudaHost.h"

inline vector<string> &split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while(getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


inline vector<string> split(const string &s, char delim) {
    vector<string> elems;
    return split(s, delim, elems);
}

uint32_t cmdParser(vector<string> args, string &host, string &port, string &apiCred, bool &debug, vector<int32_t> &devices ) {
	bool hostSet = false;
	bool apiSet = false;
	
	for (uint32_t i=0; i<args.size(); i++) {
		if (args[i][0] == '-') {
			if ((args[i].compare("-h")  == 0) || (args[i].compare("--help")  == 0)) {
				return 0x4;
			}

			if (args[i].compare("--server")  == 0) {
				if (i+1 < args.size()) {
					vector<string> tmp = split(args[i+1], ':');
					if (tmp.size() == 2) {
						host = tmp[0];
						port = tmp[1];
						hostSet = true;	
						i++;
						continue;
					}
				}
			}

			if (args[i].compare("--key")  == 0) {
				if (i+1 < args.size()) {
					apiCred = args[i+1];
					apiSet = true;
					i++;
					continue;
				}
			}

			if (args[i].compare("--devices")  == 0) {
				if (i+1 < args.size()) {
					vector<string> tmp = split(args[i+1], ',');
					for (int j=0; j<tmp.size(); j++) {
						devices.push_back(stoi(tmp[j]));
					}
					continue;
				}
			}

			if (args[i].compare("--debug")  == 0) {
				debug = true;
			}
		}
	}

	uint32_t result = 0;
	if (!hostSet) result += 1;
	if (!apiSet) result += 2;

	sort(devices.begin(), devices.end());
	
	return result;
}

int main(int argc, char* argv[]) {

	cout << "====================================" << endl;
	cout << "  BEAM Equihash 150/5 CUDA miner  " << endl;
	cout << "====================================" << endl;

	vector<string> cmdLineArgs(argv, argv+argc);
	string host;
	string port;
	string apiCred;
	bool debug = false;
	bool cpuMine = false;
	bool useTLS = true;
	vector<int32_t> devices;

	uint32_t parsing = cmdParser(cmdLineArgs, host, port, apiCred, debug, devices);

	if (parsing != 0) {
		if (parsing & 0x1) {
			cout << "Error: Parameter --server missing" << endl;
		}

		if (parsing & 0x2) {
			cout << "Error: Parameter --key missing" << endl;
		}

		cout << "Parameters: " << endl;
		cout << " --help / -h 			Showing this message" << endl;
		cout << " --server <server>:<port>	The BEAM stratum server and port to connect to (required)" << endl;
		cout << " --key <key>			The BEAM stratum server API key (required)" << endl;
		cout << " --devices <numbers>		A comma seperated list of devices that should be used for mining (default: all in system)" << endl; 
		exit(0);
	}

	beamMiner::beamStratum myStratum(host, port, apiCred, debug);

	beamMiner::CudaHost myCudaHost;
	
	cout << endl;
	cout << "Waiting for work from stratum:" << endl;
	cout << "==============================" << endl;

	myStratum.startWorking();

	while (!myStratum.hasWork()) {
		this_thread::sleep_for(std::chrono::milliseconds(200));
	}

	cout << endl;
	cout << "Start mining:" << endl;
	cout << "=============" << endl;
	myCudaHost.startMining(&myStratum, devices);
}

#if defined(_MSC_VER) && (_MSC_VER >= 1900)

FILE _iob[] = { *stdin, *stdout, *stderr };
extern "C" FILE * __cdecl __iob_func(void) { return _iob; }

#endif
