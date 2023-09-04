/*
	F5 key		become chimp
*/

#include "script.h"
#include "keyboard.h"

#include <string>
#include <ctime>

#pragma warning(disable : 4244 4305) // double <-> float conversions

void make_chimp();
void make_dude();
void check_player_model();

void make_chimp()
{
	DWORD model = GAMEPLAY::GET_HASH_KEY((char *)"a_c_chimp");
	STREAMING::REQUEST_MODEL(model);				
	while (!STREAMING::HAS_MODEL_LOADED(model))
	{
		WAIT(0);
	} 
	PLAYER::SET_PLAYER_MODEL(PLAYER::PLAYER_ID(), model);
	PED::SET_PED_DEFAULT_COMPONENT_VARIATION(PLAYER::PLAYER_PED_ID());				
	WAIT(0);
	for (int i = 0; i < 12; i++)
	{
		for (int j = 0; j < 100; j++)
		{
			int drawable = rand() % 10;
			int texture = rand() % 10;
			if (PED::IS_PED_COMPONENT_VARIATION_VALID(PLAYER::PLAYER_PED_ID(), i, drawable, texture))			
			{
				PED::SET_PED_COMPONENT_VARIATION(PLAYER::PLAYER_PED_ID(), i, drawable, texture, 0);
				break;
			}
		}
	}
	WAIT(100);
	STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(model);
			
}

void make_dude()
{
	DWORD model = GAMEPLAY::GET_HASH_KEY((char *)"player_zero");
	STREAMING::REQUEST_MODEL(model);				
	while (!STREAMING::HAS_MODEL_LOADED(model))
	{
		WAIT(0);
	} 
	PLAYER::SET_PLAYER_MODEL(PLAYER::PLAYER_ID(), model);
	PED::SET_PED_DEFAULT_COMPONENT_VARIATION(PLAYER::PLAYER_PED_ID());				
	WAIT(0);
	for (int i = 0; i < 12; i++)
	{
		for (int j = 0; j < 100; j++)
		{
			int drawable = rand() % 10;
			int texture = rand() % 10;
			if (PED::IS_PED_COMPONENT_VARIATION_VALID(PLAYER::PLAYER_PED_ID(), i, drawable, texture))			
			{
				PED::SET_PED_COMPONENT_VARIATION(PLAYER::PLAYER_PED_ID(), i, drawable, texture, 0);
				break;
			}
		}
	}
	WAIT(100);
	STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(model);
			
}

// player model control, switching on normal ped model when needed	
void check_player_model() 
{
	// common variables
	Player player = PLAYER::PLAYER_ID();
	Ped playerPed = PLAYER::PLAYER_PED_ID();	

	if (!ENTITY::DOES_ENTITY_EXIST(playerPed)) return;

	Hash model = ENTITY::GET_ENTITY_MODEL(playerPed);
	if (ENTITY::IS_ENTITY_DEAD(playerPed) || PLAYER::IS_PLAYER_BEING_ARRESTED(player, TRUE))
		if (model != GAMEPLAY::GET_HASH_KEY("player_zero") && 
			model != GAMEPLAY::GET_HASH_KEY("player_one") &&
			model != GAMEPLAY::GET_HASH_KEY("player_two"))
		{
			WAIT(1000);

			model = GAMEPLAY::GET_HASH_KEY("player_zero");
			STREAMING::REQUEST_MODEL(model);
			while (!STREAMING::HAS_MODEL_LOADED(model))
				WAIT(0);
			PLAYER::SET_PLAYER_MODEL(PLAYER::PLAYER_ID(), model);
			PED::SET_PED_DEFAULT_COMPONENT_VARIATION(PLAYER::PLAYER_PED_ID());
			STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(model);

			// wait until player is ressurected
			while (ENTITY::IS_ENTITY_DEAD(PLAYER::PLAYER_PED_ID()) || PLAYER::IS_PLAYER_BEING_ARRESTED(player, TRUE))
				WAIT(0);

		}
}

int main()
{	
	bool skinchanger_used =	false;
	bool is_chimp =	false;
	while (true)
	{
		bool press = IsKeyJustUp(VK_F5);
		if (press)
		{
			// Toggle Model
			if(is_chimp)
			{	
				make_dude();
				skinchanger_used = true;
				is_chimp = false;
			}
			else
			{
				make_chimp();
				skinchanger_used = true;
				is_chimp = true;
			}
		}
		// Check Model
		if(skinchanger_used)
		{
			check_player_model();
		}
		WAIT(0);
	}

	return 0;
}

void ScriptMain()
{
	srand(GetTickCount());
	main();
}
