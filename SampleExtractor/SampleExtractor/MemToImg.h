#pragma once
#include "def.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

class M2I {
private:
	Dumper theDumper;
	string name;
	string VMwarePath;
	binBuf buffer;
	bool VMorFILE; // true for VM, false for file
	vector<string> file_list;

	//void startVM()const;
	//void readEFL(string);
	string binToImg(string)const;
	string binToFile() const;
	void getDiffImg(string, string) const;
	void getDeltaImg(string, string);
	void getSmallImg(string, string) const;

public:
	M2I(string filename);
	M2I(string name, string path) : name(name), VMwarePath(path), theDumper(name, path), VMorFILE(true) {};
	void getIMG();
	void getIMGByCmd();
	void getDataByCmd();
	//~M2I() {
	//	for (auto i : file_list)
	//		remove(i.c_str());
	//}
	//~M2I() {
	//	delete buffer->buf;
	//	delete buffer;
	//}
};

M2I::M2I(string filename) {
	fstream f(filename, ios::in | ios::binary);
	buffer = new struct _binBuf;
	f.seekg(0, ios::end);
	buffer->bufsize = f.tellg();
	buffer->buf = new char[buffer->bufsize];
	f.seekg(0, ios::beg);
	f.read(buffer->buf, buffer->bufsize);
	f.close();
	VMorFILE = false;
	name = filename;
}

//void M2I::startVM() const {
//	theDumper.start_vm();
//}
//
//void M2I::readEFL(string EFLname) {
//	buffer = outputEFL(theDumper.dump_mem());
//}

string M2I::binToFile() const {
	int width, height;
	width = int(sqrt(buffer->bufsize));
	height = int(buffer->bufsize / width);

	Mat img(height, width, CV_8UC1, buffer->buf);

	string fileName = DATAFILEPATH + name + ".xml";
	FileStorage fs(fileName, FileStorage::WRITE);
	fs << "mem" << img;
	fs.release();
	cout << dec << "width: " << img.size().width << "\theight: " << img.size().height << endl;
	img.release();
	return fileName;
}

// 将二进制数据保存为图片
string M2I::binToImg(string imgName) const{
	// 1. 如果保存为各种格式的图像，32/16位保存方式好像都会压缩数据？？bmp
	// 2. 3通道、32位的保存方式增加了前后12个字节的数据的关联性，在后面的网络训练中需要注意
	// 3. 如果无法解决可以考虑保存为xml的格式？不用imwrit
	// 4. 注意官方文档中imwrite写了对Jpeg、png只支持16位整型，实际确实用16SC3保存的bmp是1G，32SC3保存的bmp是512M，内存都是2G，如果尝试
	// 8SC3就会触发imwrite的mat大小限制assert
	int width, height;
	width = int(sqrt(buffer->bufsize / 12));
	height = int(buffer->bufsize / 12 / width);

	Mat img(height, width, CV_32SC3, buffer->buf);

	//string imgName = DATAFILEPATH + name + ".png";
	//vector<int> compression_params;
	//compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	//compression_params.push_back(0);
	//imwrite(imgName, img, compression_params);

	/*string imgName = DATAFILEPATH + name + ".bmp";*/
	imwrite(imgName, img);
	cout << dec << "width: " << img.size().width << "\theight: " << img.size().height << endl;
	img.release();
	return imgName;
}

void M2I::getSmallImg(string imgName, string sName) const {
	// 以下方法会因为深度在resize报错
	//int width, height;
	//width = int(sqrt(buffer->bufsize / 12));
	//height = int(buffer->bufsize / 12 / width);

	//Mat img(height, width, CV_16SC3, buffer->buf);
	//Mat smallImg(416, 416, CV_16SC3);
	//resize(img, smallImg, Size(416, 416), 0, 0, INTER_AREA);

	//imwrite(sName, smallImg);
	//img.release();
	//smallImg.release();
	IplImage *src = cvLoadImage(imgName.c_str());
	IplImage *dst = cvCreateImage(CvSize(416, 416), src->depth, src->nChannels);
	cvResize(src, dst, CV_INTER_AREA);

	Mat sv = cvarrToMat(dst);
	imwrite(sName.c_str(), sv);
	// 直接cvSaveImage好像会因为opencv原因蜜汁报错
	/*cvSaveImage(sName.c_str(), dst);*/

	sv.release();
	cvReleaseImage(&src);
	cvReleaseImage(&dst);
}

void M2I::getIMG() {
	if (VMorFILE) {
		string efl = DATAFILEPATH + name + ".efl";
		theDumper.start_vm();
		getchar();
		efl = theDumper.dump_mem("");
		buffer = outputEFL(efl);
	}
	string imgName = DATAFILEPATH + name + ".bmp";
	cout << binToImg(imgName) << endl;
	// cout << binToFile() << endl;
	// cout << dec << "bufsize: " << buffer->bufsize / 1024 / 1024 << "M byte" << endl;
}

void M2I::getIMGByCmd() {
	if (!VMorFILE)
		assert("error");
	theDumper.start_vm();

	while (1) {
		string input;
		cin >> input;
		if (input == "Q")
			break;
		string efl = theDumper.dump_mem(input);
		file_list.push_back(efl);
		buffer = outputEFL(efl);
		//string outputName = efl + ".Bin";
		//file_list.push_back(outputName);
		string imgName = DATAFILEPATH + name + "_" + input + ".bmp";
		cout << binToImg(imgName) << endl;
		file_list.push_back(imgName);
	}
}

// 和基准图作差后求surf特征点，将特征点输出到.csv的文本文件中
void M2I::getDiffImg(string initImg, string input) const {
	string difFile = DATAFILEPATH;
	difFile += "diff_" + input + ".bmp";
	IplImage *orig, *init, *diff;
	//string orig_s = DATAFILEPATH + name + ".bmp";
	string orig_s = DATAFILEPATH; orig_s += "benchmark.bmp";
	orig = cvLoadImage(orig_s.c_str());
	init = cvLoadImage(initImg.c_str());
	diff = cvCreateImage(cvSize(orig->width, orig->height), orig->depth, orig->nChannels);
	cvAbsDiff(init, orig, diff);
	Mat dif = cvarrToMat(diff);

	imwrite("tmp.bmp", dif);
	int Hessian = 18000;
	vector<KeyPoint> KP = getCudaSURF("tmp.bmp", Hessian);

	//int minHessian = 16000;
	//Mat marked;
	//Ptr<SURF> detector = SURF::create(minHessian);
	//vector<KeyPoint> keypoints;
	//cout << "detecting SURF..." << endl;
	//detector->detect(dif, keypoints, Mat());
	////cout << "drawing keypoints..." << endl;
	////drawKeypoints(dif, keypoints, marked);
	////cout << "writing into disk..." << endl;
	////imwrite("surf points marked.bmp", marked);
	////cout << "# of keypoints: " << keypoints.size() << endl;
	////imwrite("original diff.bmp", dif);
	//marked.release();

	string csv_s = DATAFILEPATH + name + input + ".csv";
	ofstream csv(csv_s);
	csv << "x,y,size,response" << endl;
	for (auto i: KP)
		csv << i.pt.x << ',' << i.pt.y << ',' << i.size << ',' << i.response << endl;
	
	csv.close();
	dif.release();
	cvReleaseImage(&orig);
	cvReleaseImage(&init);
	cvReleaseImage(&diff);
}

// 生成xdelta3差分文件, resize到6000*2000的灰度图
void M2I::getDeltaImg(string initImg, string input) {
	//string delta = "xdelta3 -v -e -s ";
	string delta = "xdelta3 -e -s ";
	string difFile = "xdelta_" + input;
	//delta += DATAFILEPATH + name + ".bmp " + initImg + " " + DATAFILEPATH + "xdelta_" + input;
	delta += DATAFILEPATH;
	delta += "benchmark.bmp " + initImg + " " + DATAFILEPATH + "xdelta_" + input;
	cout << delta << endl;
	system(delta.c_str());

	string filename = DATAFILEPATH + difFile;
	file_list.push_back(filename);
	fstream f(filename, ios::in | ios::binary);
	binBuf dif_buffer = new struct _binBuf;
	f.seekg(0, ios::end);
	dif_buffer->bufsize = f.tellg();
	dif_buffer->buf = new char[dif_buffer->bufsize];
	f.seekg(0, ios::beg);
	f.read(dif_buffer->buf, dif_buffer->bufsize);
	f.close();

	string imgName = filename + "_" + name + ".bmp";
	int width, height;
	width = int(sqrt(dif_buffer->bufsize / 1));
	height = int(dif_buffer->bufsize / 1 / width);
	Mat img(height, width, CV_8UC1, dif_buffer->buf);
	//imwrite("last_ini.bmp", img);
	Mat dst(6000, 2000, CV_8UC1);
	cv::resize(img, dst, Size(6000, 2000), 0, 0, INTER_AREA);

	imwrite(imgName, dst);

	cout << imgName << ": initial size: " << dif_buffer->bufsize << endl;
	img.release();
	dst.release();
	delete dif_buffer->buf;
	delete dif_buffer;
}

void M2I::getDataByCmd() {
	if (!VMorFILE)
		assert("error");
	theDumper.start_vm();

	while (1) {
		string input;
		cin >> input;

		clock_t start, end, mid;
		start = mid =clock();

		if (input == "Q")
			break;
		// dump内存，获得EFL文件
		string efl = theDumper.dump_mem(input);

		// 这种操作是在保存大文件的名称以便每个循环结束删除中间文件
		file_list.push_back(efl);

		// 处理EFL文件生成图片
		buffer = outputEFL(efl);
		string imgName = DATAFILEPATH + name + "__" + input + ".bmp";
		string smallImgName = DATAFILEPATH + name + "_" + input + ".bmp";
		binToImg(imgName);
		file_list.push_back(imgName);

		double duration;
		end = clock();
		duration = (double)(end - mid) / CLOCKS_PER_SEC;
		cout << "time elapsed for pumping: " << duration << endl;
		mid = end;

		getDeltaImg(imgName, input);
		end = clock();
		duration = (double)(end - mid) / CLOCKS_PER_SEC;
		cout << "time elapsed for delta: " << duration << endl;
		mid = end;

		getDiffImg(imgName, input);
		end = clock();
		duration = (double)(end - mid) / CLOCKS_PER_SEC;
		cout << "time elapsed for diff&surf: " << duration << endl;
		mid = end;

		getSmallImg(imgName, smallImgName);
		end = clock();
	    duration = (double)(end - mid) / CLOCKS_PER_SEC;
		cout << "time elapsed for down sampled image: " << duration << endl;
		duration = (double)(end - start) / CLOCKS_PER_SEC;
		cout << "time elapsed for input \'" << input << "\': " << duration << endl;

		for (auto i : file_list)
			remove(i.c_str());
		file_list.clear();
		delete buffer->buf;
		delete buffer;
	}
}