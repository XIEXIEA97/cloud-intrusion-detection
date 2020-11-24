// '不是自己写的'读取EFL格式内存并转化为二进制格式数据的函数
#pragma once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <windows.h>
#include <iostream>
#include <string>
using namespace std;
typedef UINT64 Elf64_Addr;
typedef UINT64 Elf64_Off;
typedef UINT16 Elf64_Half;
typedef UINT32 Elf64_Word;
typedef INT32 Elf64_Sword;
typedef UINT64 Elf64_Xword;
typedef INT64 Elf_Sxword;
#define PT_LOAD 1
#define PT_NOTE 0
struct ELFheader {
	unsigned char e_ident[16];
	Elf64_Half e_type;
	Elf64_Half e_machine;
	Elf64_Word e_version;
	Elf64_Addr e_entry;
	Elf64_Off e_phoff;
	Elf64_Off e_shorff;
	Elf64_Word e_flags;
	Elf64_Half e_ehsize;
	Elf64_Half e_phentsize;
	Elf64_Half e_phnum;
	Elf64_Half e_shentsize;
	Elf64_Half e_shnum;
	Elf64_Half e_shstrndx;
};

struct ELFPheader {
	Elf64_Word p_type;
	Elf64_Word p_flags;
	Elf64_Off p_offset;
	Elf64_Addr p_vaddr;
	Elf64_Addr p_paddr;
	Elf64_Xword p_filesz;
	Elf64_Xword p_memsz;
	Elf64_Xword p_align;
};

typedef struct _binBuf *binBuf;

struct _binBuf {
	UINT64 bufsize;
	char* buf;
};

binBuf outputEFL(string ELF_Filename)
{
	binBuf ret = NULL;
	HANDLE hFile = INVALID_HANDLE_VALUE;
	HANDLE hMap = NULL;
	LPVOID pFile = NULL;
	string outputName = ELF_Filename + ".Bin";
	hFile = CreateFile(ELF_Filename.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_ARCHIVE, 0);

	if (hFile == INVALID_HANDLE_VALUE)
	{
		printf("OpenFile Error,Code:%d\n", GetLastError());
		system("pause");
		return NULL;
	}

	hMap = CreateFileMapping(hFile, NULL, PAGE_READWRITE, NULL, NULL, NULL);
	if (hMap == NULL)
	{
		printf("CreateFileMapping Error,Code:%d\n", GetLastError());
		CloseHandle(hFile);
		system("pause");
		return NULL;
	}

	pFile = MapViewOfFile(hMap, FILE_MAP_READ | FILE_MAP_WRITE, NULL, NULL, NULL);
	if (pFile == NULL)
	{
		printf("MapViewOfFile Error,Code:%d\n", GetLastError());
		CloseHandle(hMap);
		CloseHandle(hFile);
		system("pause");
		return NULL;
	}
	ret = new struct _binBuf;
	//HANDLE rawMem = CreateFile(outputName.c_str(), GENERIC_WRITE | GENERIC_READ, 0, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
	ELFheader* p_Elf64_header = (ELFheader*)pFile;
	DWORD dwWritenSize = 0;
	cout << hex << p_Elf64_header->e_ehsize << '\t' << p_Elf64_header->e_phentsize << endl;
	for (UINT64 i = 0; i < p_Elf64_header->e_phnum; i++)
	{
		ELFPheader *p_prheader = (ELFPheader *)((UINT64)pFile + i * p_Elf64_header->e_phentsize + p_Elf64_header->e_ehsize);
		if (p_prheader->p_type == PT_LOAD)
		{
			UINT64 index = (UINT64)pFile + p_prheader->p_offset;
			ret->bufsize = p_prheader->p_filesz;
			ret->buf = new char[ret->bufsize];
			memcpy(ret->buf, (void *)(index), p_prheader->p_filesz);
			cout << hex << p_prheader->p_offset << '\t' << p_prheader->p_filesz << endl;
			goto end;
		}
	}
end:
	UnmapViewOfFile(pFile);
	CloseHandle(hMap);
	CloseHandle(hFile);
	//remove(ELF_Filename.c_str());
	//remove(outputName.c_str());
	return ret;
}