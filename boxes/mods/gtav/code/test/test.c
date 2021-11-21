// Test ASI
#include <windows.h>
#include "natives.h" 
#define IMPORT __declspec(dllimport)

BOOL APIENTRY DllMain(HMODULE hInstance, DWORD reason, LPVOID lpReserved)
{
	switch (reason)
	{
	case DLL_PROCESS_ATTACH:
		break;
	case DLL_PROCESS_DETACH:
		break;
	}		
	return TRUE;
}