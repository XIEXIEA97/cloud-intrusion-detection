#include "MemToImg.h"

int main(int argc, char* argv[]) {

	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    //testCudaSURF();

	//M2I m2i("Agent", "cd c:\\program files\\oracle\\virtualbox ");
	//M2I m2i("Lady", "cd c:\\program files\\oracle\\virtualbox ");
	//M2I m2i("Mayday", "cd c:\\program files\\oracle\\virtualbox ");
	//M2I m2i("Dofloo", "cd c:\\program files\\oracle\\virtualbox ");
	//M2I m2i("Ganiw", "cd c:\\program files\\oracle\\virtualbox ");
	M2I m2i("Ubuntu1704", "cd c:\\program files\\oracle\\virtualbox ");
	//M2I m2i("chrome.dll");
	//m2i.getIMG();
	//m2i.getIMGByCmd();
	m2i.getDataByCmd();
	system("pause");
	return 0;

}