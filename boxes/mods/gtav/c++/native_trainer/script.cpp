/*
	THIS FILE IS A PART OF GTA V SCRIPT HOOK SDK
				http://dev-c.com			
			(C) Alexander Blade 2015
*/

/*
	F4					activate
	NUM2/8/4/6			navigate thru the menus and lists (numlock must be on)
	NUM5 				select
	NUM0/BACKSPACE/F4 	back
	NUM9/3 				use vehicle boost when active
	NUM+ 				use vehicle rockets when active
*/

#include "script.h"
#include "keyboard.h"

#include <string>
#include <ctime>

#pragma warning(disable : 4244 4305) // double <-> float conversions

void draw_rect(float A_0, float A_1, float A_2, float A_3, int A_4, int A_5, int A_6, int A_7)
{
	GRAPHICS::DRAW_RECT((A_0 + (A_2 * 0.5f)), (A_1 + (A_3 * 0.5f)), A_2, A_3, A_4, A_5, A_6, A_7);
}

void draw_menu_line(std::string caption, float lineWidth, float lineHeight, float lineTop, float lineLeft, float textLeft, bool active, bool title, bool rescaleText = true)
{
	// default values
	int text_col[4] = {255, 255, 255, 255},
		rect_col[4] = {70, 95, 95, 255};
	float text_scale = 0.35;
	int font = 0;

	// correcting values for active line
	if (active)
	{
		text_col[0] = 0;
		text_col[1] = 0;
		text_col[2] = 0;

		rect_col[0] = 218;
		rect_col[1] = 242;
		rect_col[2] = 216;

		if (rescaleText) text_scale = 0.40;
	}

	if (title)
	{
		rect_col[0] = 0;
		rect_col[1] = 0;
		rect_col[2] = 0;

		if (rescaleText) text_scale = 0.50;
		font = 1;
	}

	int screen_w, screen_h;
	GRAPHICS::GET_SCREEN_RESOLUTION(&screen_w, &screen_h);

	textLeft += lineLeft;

	float lineWidthScaled = lineWidth / (float)screen_w; // line width
	float lineTopScaled = lineTop / (float)screen_h; // line top offset
	float textLeftScaled = textLeft / (float)screen_w; // text left offset
	float lineHeightScaled = lineHeight / (float)screen_h; // line height

	float lineLeftScaled = lineLeft / (float)screen_w;

	// this is how it's done in original scripts

	// text upper part
	UI::SET_TEXT_FONT(font);
	UI::SET_TEXT_SCALE(0.0, text_scale);
	UI::SET_TEXT_COLOUR(text_col[0], text_col[1], text_col[2], text_col[3]);
	UI::SET_TEXT_CENTRE(0);
	UI::SET_TEXT_DROPSHADOW(0, 0, 0, 0, 0);
	UI::SET_TEXT_EDGE(0, 0, 0, 0, 0);
	UI::_SET_TEXT_ENTRY("STRING");
	UI::_ADD_TEXT_COMPONENT_STRING((LPSTR)caption.c_str());
	UI::_DRAW_TEXT(textLeftScaled, (((lineTopScaled + 0.00278f) + lineHeightScaled) - 0.005f));

	// text lower part
	UI::SET_TEXT_FONT(font);
	UI::SET_TEXT_SCALE(0.0, text_scale);
	UI::SET_TEXT_COLOUR(text_col[0], text_col[1], text_col[2], text_col[3]);
	UI::SET_TEXT_CENTRE(0);
	UI::SET_TEXT_DROPSHADOW(0, 0, 0, 0, 0);
	UI::SET_TEXT_EDGE(0, 0, 0, 0, 0);
	UI::_SET_TEXT_GXT_ENTRY("STRING");
	UI::_ADD_TEXT_COMPONENT_STRING((LPSTR)caption.c_str());
	int num25 = UI::_0x9040DFB09BE75706(textLeftScaled, (((lineTopScaled + 0.00278f) + lineHeightScaled) - 0.005f));

	// rect
	draw_rect(lineLeftScaled, lineTopScaled + (0.00278f), 
		lineWidthScaled, ((((float)(num25)* UI::_0xDB88A37483346780(text_scale, 0)) + (lineHeightScaled * 2.0f)) + 0.005f),
		rect_col[0], rect_col[1], rect_col[2], rect_col[3]);	
}

bool trainer_switch_pressed()
{
	return IsKeyJustUp(VK_F4);
}

void get_button_state(bool *a, bool *b, bool *up, bool *down, bool *l, bool *r)
{
	if (a) *a = IsKeyDown(VK_NUMPAD5);
	if (b) *b = IsKeyDown(VK_NUMPAD0) || trainer_switch_pressed() || IsKeyDown(VK_BACK);
	if (up) *up = IsKeyDown(VK_NUMPAD8);
	if (down) *down = IsKeyDown(VK_NUMPAD2);
	if (r) *r = IsKeyDown(VK_NUMPAD6);
	if (l) *l = IsKeyDown(VK_NUMPAD4);
}

void menu_beep()
{
	AUDIO::PLAY_SOUND_FRONTEND(-1, "NAV_UP_DOWN", "HUD_FRONTEND_DEFAULT_SOUNDSET", 0);
}

std::string statusText;
DWORD statusTextDrawTicksMax;
bool statusTextGxtEntry;

void update_status_text()
{
	if (GetTickCount() < statusTextDrawTicksMax)
	{
		UI::SET_TEXT_FONT(0);
		UI::SET_TEXT_SCALE(0.55, 0.55);
		UI::SET_TEXT_COLOUR(255, 255, 255, 255);
		UI::SET_TEXT_WRAP(0.0, 1.0);
		UI::SET_TEXT_CENTRE(1);
		UI::SET_TEXT_DROPSHADOW(0, 0, 0, 0, 0);
		UI::SET_TEXT_EDGE(1, 0, 0, 0, 205);
		if (statusTextGxtEntry)
		{
			UI::_SET_TEXT_ENTRY((char *)statusText.c_str());
		} else
		{
			UI::_SET_TEXT_ENTRY("STRING");
			UI::_ADD_TEXT_COMPONENT_STRING((char *)statusText.c_str());
		}
		UI::_DRAW_TEXT(0.5, 0.5);
	}
}

void set_status_text(std::string str, DWORD time = 2500, bool isGxtEntry = false)
{
	statusText = str;
	statusTextDrawTicksMax = GetTickCount() + time;
	statusTextGxtEntry = isGxtEntry;
}

// features
bool featurePlayerInvincible			=	false;
bool featurePlayerInvincibleUpdated		=	false;
bool featurePlayerNeverWanted			=	false;
bool featurePlayerIgnored				=	false;
bool featurePlayerIgnoredUpdated		=	false;
bool featurePlayerUnlimitedAbility		=	false;
bool featurePlayerNoNoise				=	false;
bool featurePlayerNoNoiseUpdated		=	false;
bool featurePlayerFastSwim				=	false;
bool featurePlayerFastSwimUpdated		=	false;
bool featurePlayerFastRun				=	false;
bool featurePlayerFastRunUpdated		=	false;
bool featurePlayerSuperJump				=	false;

bool featureWeaponNoReload				=	false;
bool featureWeaponFireAmmo				=	false;
bool featureWeaponExplosiveAmmo			=	false;
bool featureWeaponExplosiveMelee		=	false;
bool featureWeaponVehRockets			=	false;

DWORD featureWeaponVehShootLastTime		=	0;

bool featureVehInvincible				=	false;
bool featureVehInvincibleUpdated		=	false;
bool featureVehInvincibleWheels			=	false;
bool featureVehInvincibleWheelsUpdated	=	false;
bool featureVehSeatbelt					=	false;
bool featureVehSeatbeltUpdated			=	false;
bool featureVehSpeedBoost				=	false;
bool featureVehWrapInSpawned			=	false;

bool featureWorldMoonGravity			=	false;
bool featureWorldRandomCops				=	true;
bool featureWorldRandomTrains			=	true;
bool featureWorldRandomBoats			=	true;
bool featureWorldGarbageTrucks			=	true;

bool featureTimePaused					=	false;
bool featureTimePausedUpdated			=	false;
bool featureTimeSynced					=	false;

bool featureWeatherWind					=	false;
bool featureWeatherPers					=	false;

bool featureMiscLockRadio				=	false;
bool featureMiscHideHud					=	false;


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
			set_status_text("turning to normal");
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

void update_vehicle_guns()
{
	Player player = PLAYER::PLAYER_ID();
	Ped playerPed = PLAYER::PLAYER_PED_ID();	

	if (!ENTITY::DOES_ENTITY_EXIST(playerPed) || !featureWeaponVehRockets) return;

	bool bSelect = IsKeyDown(0x6B); // num plus
	if (bSelect && featureWeaponVehShootLastTime + 150 < GetTickCount() &&
		PLAYER::IS_PLAYER_CONTROL_ON(player) &&	PED::IS_PED_IN_ANY_VEHICLE(playerPed, 0))
	{
		Vehicle veh = PED::GET_VEHICLE_PED_IS_USING(playerPed);

		Vector3 v0, v1;
		GAMEPLAY::GET_MODEL_DIMENSIONS(ENTITY::GET_ENTITY_MODEL(veh), &v0, &v1);

		Hash weaponAssetRocket = GAMEPLAY::GET_HASH_KEY("WEAPON_VEHICLE_ROCKET");
		if (!WEAPON::HAS_WEAPON_ASSET_LOADED(weaponAssetRocket))
		{
			WEAPON::REQUEST_WEAPON_ASSET(weaponAssetRocket, 31, 0);
			while (!WEAPON::HAS_WEAPON_ASSET_LOADED(weaponAssetRocket))
				WAIT(0);
		}

		Vector3 coords0from = ENTITY::GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(veh, -(v1.x + 0.25f), v1.y + 1.25f, 0.1);
		Vector3 coords1from = ENTITY::GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(veh,  (v1.x + 0.25f), v1.y + 1.25f, 0.1);
		Vector3 coords0to = ENTITY::GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(veh, -v1.x, v1.y + 100.0f, 0.1f);
		Vector3 coords1to = ENTITY::GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(veh,  v1.x, v1.y + 100.0f, 0.1f);

		GAMEPLAY::SHOOT_SINGLE_BULLET_BETWEEN_COORDS(coords0from.x, coords0from.y, coords0from.z, 
													 coords0to.x, coords0to.y, coords0to.z, 
													 250, 1, weaponAssetRocket, playerPed, 1, 0, -1.0);
		GAMEPLAY::SHOOT_SINGLE_BULLET_BETWEEN_COORDS(coords1from.x, coords1from.y, coords1from.z, 
													 coords1to.x, coords1to.y, coords1to.z, 
													 250, 1, weaponAssetRocket, playerPed, 1, 0, -1.0);

		featureWeaponVehShootLastTime = GetTickCount();
	}
}

bool skinchanger_used = false;

// Updates all features that can be turned off by the game, being called each game frame
void update_features() 
{
	update_status_text();

	update_vehicle_guns();

	// changing player model if died/arrested while being in another skin, since it can cause inf loading loop
	if (skinchanger_used)
		check_player_model();

	// wait until player is ready, basicly to prevent using the trainer while player is dead or arrested
	while (ENTITY::IS_ENTITY_DEAD(PLAYER::PLAYER_PED_ID()) || PLAYER::IS_PLAYER_BEING_ARRESTED(PLAYER::PLAYER_ID(), TRUE))
		WAIT(0);

	// read default feature values from the game
	featureWorldRandomCops = PED::CAN_CREATE_RANDOM_COPS() == TRUE;

	// common variables
	Player player = PLAYER::PLAYER_ID();
	Ped playerPed = PLAYER::PLAYER_PED_ID();	
	BOOL bPlayerExists = ENTITY::DOES_ENTITY_EXIST(playerPed);

	// player invincible
	if (featurePlayerInvincibleUpdated)
	{
		if (bPlayerExists && !featurePlayerInvincible)
			PLAYER::SET_PLAYER_INVINCIBLE(player, FALSE);
		featurePlayerInvincibleUpdated = false;
	}
	if (featurePlayerInvincible)
	{
		if (bPlayerExists)
			PLAYER::SET_PLAYER_INVINCIBLE(player, TRUE);
	}

	// player never wanted
	if (featurePlayerNeverWanted)
	{
		if (bPlayerExists)
			PLAYER::CLEAR_PLAYER_WANTED_LEVEL(player);
	}

	// police ignore player
	if (featurePlayerIgnoredUpdated)
	{
		if (bPlayerExists)
			PLAYER::SET_POLICE_IGNORE_PLAYER(player, featurePlayerIgnored);
		featurePlayerIgnoredUpdated = false;
	}

	// player special ability
	if (featurePlayerUnlimitedAbility)
	{
		if (bPlayerExists)
			PLAYER::SPECIAL_ABILITY_FILL_METER(player, 1);
	}

	// player no noise
	if (featurePlayerNoNoiseUpdated)
	{
		if (bPlayerExists && !featurePlayerNoNoise)
			PLAYER::SET_PLAYER_NOISE_MULTIPLIER(player, 1.0);
		featurePlayerNoNoiseUpdated = false;
	}
	if (featurePlayerNoNoise)
		PLAYER::SET_PLAYER_NOISE_MULTIPLIER(player, 0.0);

	// player fast swim
	if (featurePlayerFastSwimUpdated)
	{
		if (bPlayerExists && !featurePlayerFastSwim)
			PLAYER::SET_SWIM_MULTIPLIER_FOR_PLAYER(player, 1.0);
		featurePlayerFastSwimUpdated = false;
	}
	if (featurePlayerFastSwim)
		PLAYER::SET_SWIM_MULTIPLIER_FOR_PLAYER(player, 1.49);

	// player fast run
	if (featurePlayerFastRunUpdated)
	{
		if (bPlayerExists && !featurePlayerFastRun)
			PLAYER::SET_RUN_SPRINT_MULTIPLIER_FOR_PLAYER(player, 1.0);
		featurePlayerFastRunUpdated = false;
	}
	if (featurePlayerFastRun)
		PLAYER::SET_RUN_SPRINT_MULTIPLIER_FOR_PLAYER(player, 1.49);

	// player super jump
	if (featurePlayerSuperJump)
	{
		if (bPlayerExists)
			GAMEPLAY::SET_SUPER_JUMP_THIS_FRAME(player);
	}

	// weapon
	if (featureWeaponFireAmmo)
	{
		if (bPlayerExists)
			GAMEPLAY::SET_FIRE_AMMO_THIS_FRAME(player);
	}
	if (featureWeaponExplosiveAmmo)
	{
		if (bPlayerExists)
			GAMEPLAY::SET_EXPLOSIVE_AMMO_THIS_FRAME(player);
	}
	if (featureWeaponExplosiveMelee)
	{
		if (bPlayerExists)
			GAMEPLAY::SET_EXPLOSIVE_MELEE_THIS_FRAME(player);
	}

	// weapon no reload
	if (bPlayerExists && featureWeaponNoReload)
	{
		Hash cur;
		if (WEAPON::GET_CURRENT_PED_WEAPON(playerPed, &cur, 1))
		{
			if (WEAPON::IS_WEAPON_VALID(cur))
			{
				int maxAmmo;
				if (WEAPON::GET_MAX_AMMO(playerPed, cur, &maxAmmo))
				{
					WEAPON::SET_PED_AMMO(playerPed, cur, maxAmmo);

					maxAmmo = WEAPON::GET_MAX_AMMO_IN_CLIP(playerPed, cur, 1);
					if (maxAmmo > 0)
						WEAPON::SET_AMMO_IN_CLIP(playerPed, cur, maxAmmo);
				}
			}
		}
	}

	// player's vehicle invincible
	if (featureVehInvincibleUpdated)
	{
		if (bPlayerExists && !featureVehInvincible && PED::IS_PED_IN_ANY_VEHICLE(playerPed, 0))
		{
			Vehicle veh = PED::GET_VEHICLE_PED_IS_USING(playerPed);
			ENTITY::SET_ENTITY_INVINCIBLE(veh, FALSE);
			ENTITY::SET_ENTITY_PROOFS(veh, 0, 0, 0, 0, 0, 0, 0, 0);
			VEHICLE::SET_VEHICLE_TYRES_CAN_BURST(veh, 1);
			VEHICLE::SET_VEHICLE_WHEELS_CAN_BREAK(veh, 1);
			VEHICLE::SET_VEHICLE_CAN_BE_VISIBLY_DAMAGED(veh, 1);
		}
		featureVehInvincibleUpdated = false;
	}
	if (featureVehInvincible)
	{
		if (bPlayerExists && PED::IS_PED_IN_ANY_VEHICLE(playerPed, 0))
		{
			Vehicle veh = PED::GET_VEHICLE_PED_IS_USING(playerPed);
			ENTITY::SET_ENTITY_INVINCIBLE(veh, TRUE);
			ENTITY::SET_ENTITY_PROOFS(veh, 1, 1, 1, 1, 1, 1, 1, 1);	
			VEHICLE::SET_VEHICLE_TYRES_CAN_BURST(veh, 0);
			VEHICLE::SET_VEHICLE_WHEELS_CAN_BREAK(veh, 0);
			VEHICLE::SET_VEHICLE_CAN_BE_VISIBLY_DAMAGED(veh, 0);
		}
	}

	// player's vehicle invincible wheels, usefull with custom handling
	if (featureVehInvincibleWheelsUpdated)
	{
		if (bPlayerExists && !featureVehInvincibleWheels && PED::IS_PED_IN_ANY_VEHICLE(playerPed, 0))
		{
			Vehicle veh = PED::GET_VEHICLE_PED_IS_USING(playerPed);
			VEHICLE::SET_VEHICLE_TYRES_CAN_BURST(veh, TRUE);
			VEHICLE::SET_VEHICLE_WHEELS_CAN_BREAK(veh, TRUE);
			VEHICLE::SET_VEHICLE_HAS_STRONG_AXLES(veh, FALSE);
		}
		featureVehInvincibleWheelsUpdated = false;
	}
	if (featureVehInvincibleWheels)
	{
		if (bPlayerExists && PED::IS_PED_IN_ANY_VEHICLE(playerPed, 0))
		{
			Vehicle veh = PED::GET_VEHICLE_PED_IS_USING(playerPed);
			VEHICLE::SET_VEHICLE_TYRES_CAN_BURST(veh, FALSE);
			VEHICLE::SET_VEHICLE_WHEELS_CAN_BREAK(veh, FALSE);
			VEHICLE::SET_VEHICLE_HAS_STRONG_AXLES(veh, TRUE);
		}
	}

	// seat belt
	const int PED_FLAG_CAN_FLY_THRU_WINDSCREEN = 32;
	if (featureVehSeatbeltUpdated)
	{
		if (bPlayerExists && !featureVehSeatbelt)
			PED::SET_PED_CONFIG_FLAG(playerPed, PED_FLAG_CAN_FLY_THRU_WINDSCREEN, TRUE);
		featureVehSeatbeltUpdated = false;
	}
	if (featureVehSeatbelt)
	{
		if (bPlayerExists)
		{
			if (PED::GET_PED_CONFIG_FLAG(playerPed, PED_FLAG_CAN_FLY_THRU_WINDSCREEN, TRUE))
				PED::SET_PED_CONFIG_FLAG(playerPed, PED_FLAG_CAN_FLY_THRU_WINDSCREEN, FALSE);
		}
	}

	// player's vehicle boost
	if (featureVehSpeedBoost && bPlayerExists && PED::IS_PED_IN_ANY_VEHICLE(playerPed, 0))
	{
		Vehicle veh = PED::GET_VEHICLE_PED_IS_USING(playerPed);
		DWORD model = ENTITY::GET_ENTITY_MODEL(veh);

		bool bUp = IsKeyDown(VK_NUMPAD9);
		bool bDown = IsKeyDown(VK_NUMPAD3);

		if (bUp || bDown)
		{			
			float speed = ENTITY::GET_ENTITY_SPEED(veh);
			if (bUp) 
			{
				if (speed < 3.0f) speed = 3.0f;
				speed += speed * 0.05f;
				VEHICLE::SET_VEHICLE_FORWARD_SPEED(veh, speed);
			} else
			if (ENTITY::IS_ENTITY_IN_AIR(veh) || speed > 5.0) 
				VEHICLE::SET_VEHICLE_FORWARD_SPEED(veh, 0.0);
		}
	}

	// time pause
	if (featureTimePausedUpdated)
	{
		TIME::PAUSE_CLOCK(featureTimePaused);
		featureTimePausedUpdated = false;
	}

	// time sync
	if (featureTimeSynced)
	{
		time_t now = time(0);
		tm t;
		localtime_s(&t, &now);
		TIME::SET_CLOCK_TIME(t.tm_hour, t.tm_min, t.tm_sec);
	}

	// hide hud
	if (featureMiscHideHud)
		UI::HIDE_HUD_AND_RADAR_THIS_FRAME();
}

LPCSTR pedModels[1][2] = {
	{"player_zero", "a_c_chimp"}
};

LPCSTR pedModelNames[1][2] = {
	{"MICHAEL", "CHIMP"}
};

int skinchangerActiveLineIndex = 0;
int skinchangerActiveItemIndex = 0;

bool process_skinchanger_menu()
{
	DWORD waitTime = 150;
	const int lineCount = 1;
	const int itemCount = 2;
	const int itemCountLastLine = 2;
	while (true)
	{
		// timed menu draw, used for pause after active line switch
		DWORD maxTickCount = GetTickCount() + waitTime;
		do 
		{
			// draw menu
			char caption[32];
			sprintf_s(caption, "SKIN CHANGER   %d / %d", skinchangerActiveLineIndex + 1, lineCount);
			draw_menu_line(caption, 350.0, 15.0, 18.0, 0.0, 5.0, false, true);
			for (int i = 0; i < itemCount; i++)
				if (strlen(pedModels[skinchangerActiveLineIndex][i]) || strcmp(pedModelNames[skinchangerActiveLineIndex][i], "NONE") == 0)
					draw_menu_line(pedModelNames[skinchangerActiveLineIndex][i], 100.0f, 5.0f, 200.0f, 100.0f + i * 110.0f, 5.0f, i == skinchangerActiveItemIndex, false, false);
			
			update_features();
			WAIT(0);
		} while (GetTickCount() < maxTickCount);
		waitTime = 0;

		bool bSelect, bBack, bUp, bDown, bLeft, bRight;
		get_button_state(&bSelect, &bBack, &bUp, &bDown, &bLeft, &bRight);
		
		if (bSelect)
		{
			menu_beep();
			DWORD model = GAMEPLAY::GET_HASH_KEY((char *)pedModels[skinchangerActiveLineIndex][skinchangerActiveItemIndex]);
			if (STREAMING::IS_MODEL_IN_CDIMAGE(model) && STREAMING::IS_MODEL_VALID(model))
			{
				STREAMING::REQUEST_MODEL(model);				
				while (!STREAMING::HAS_MODEL_LOADED(model))	WAIT(0); 
				//STREAMING::LOAD_ALL_OBJECTS_NOW();
				PLAYER::SET_PLAYER_MODEL(PLAYER::PLAYER_ID(), model);
				//PED::SET_PED_RANDOM_COMPONENT_VARIATION(PLAYER::PLAYER_PED_ID(), FALSE);
				PED::SET_PED_DEFAULT_COMPONENT_VARIATION(PLAYER::PLAYER_PED_ID());				
				WAIT(0);
				for (int i = 0; i < 12; i++)
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
				skinchanger_used = true;
				WAIT(100);
				STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(model);				
				waitTime = 200;
			}
		} else
		if (bBack)
		{
			menu_beep();
			break;
		} else
		if (bRight)
		{
			menu_beep();
			skinchangerActiveItemIndex++;
			int itemsMax = (skinchangerActiveLineIndex == (lineCount - 1)) ? itemCountLastLine : itemCount;
			if (skinchangerActiveItemIndex == itemsMax) 
				skinchangerActiveItemIndex = 0;			
			waitTime = 100;
		} else
		if (bLeft)
		{
			menu_beep();
			if (skinchangerActiveItemIndex == 0) 
				skinchangerActiveItemIndex = (skinchangerActiveLineIndex == (lineCount - 1)) ? itemCountLastLine : itemCount;
			skinchangerActiveItemIndex--;
			waitTime = 100;
		} else
		if (bUp)
		{
			menu_beep();
			if (skinchangerActiveLineIndex == 0) 
				skinchangerActiveLineIndex = lineCount;
			skinchangerActiveLineIndex--;
			waitTime = 200;
		} else
		if (bDown)
		{
			menu_beep();
			skinchangerActiveLineIndex++;
			if (skinchangerActiveLineIndex == lineCount) 
				skinchangerActiveLineIndex = 0;			
			waitTime = 200;
		}
		if (skinchangerActiveLineIndex == (lineCount - 1))
			if (skinchangerActiveItemIndex >= itemCountLastLine)
				skinchangerActiveItemIndex = 0;
	}
	return false;
}

int teleportActiveLineIndex = 0;

bool process_teleport_menu()
{
	const float lineWidth = 250.0;
	const int lineCount	= 17;

	std::string caption = "TELEPORT";

	static struct {
		LPCSTR  text;
		float x;
		float y;
		float z;
	} lines[lineCount] = {
			{ "MARKER" },
			{ "MICHAEL'S HOUSE", -852.4f, 160.0f, 65.6f },
			{ "FRANKLIN'S HOUSE", 7.9f, 548.1f, 175.5f },
			{ "TREVOR'S TRAILER", 1985.7f, 3812.2f, 32.2f },
			{ "AIRPORT ENTRANCE", -1034.6f, -2733.6f, 13.8f },
			{ "AIRPORT FIELD", -1336.0f, -3044.0f, 13.9f },
			{ "ELYSIAN ISLAND", 338.2f, -2715.9f, 38.5f },
			{ "JETSAM", 760.4f, -2943.2f, 5.8f },
			{ "STRIPCLUB", 127.4f, -1307.7f, 29.2f },
			{ "ELBURRO HEIGHTS", 1384.0f, -2057.1f, 52.0f },
			{ "FERRIS WHEEL", -1670.7f, -1125.0f, 13.0f },
			{ "CHUMASH", -3192.6f, 1100.0f, 20.2f },
			{ "WINDFARM", 2354.0f, 1830.3f, 101.1f },
			{ "MILITARY BASE", -2047.4f, 3132.1f, 32.8f },
			{ "MCKENZIE AIRFIELD", 2121.7f, 4796.3f, 41.1f },
			{ "DESERT AIRFIELD", 1747.0f, 3273.7f, 41.1f },
			{ "CHILLIAD", 425.4f, 5614.3f, 766.5f }
	};

	DWORD waitTime = 150;
	while (true)
	{
		// timed menu draw, used for pause after active line switch
		DWORD maxTickCount = GetTickCount() + waitTime;
		do 
		{
			// draw menu
			draw_menu_line(caption, lineWidth, 15.0, 18.0, 0.0, 5.0, false, true);
			for (int i = 0; i < lineCount; i++)
				if (i != teleportActiveLineIndex)
					draw_menu_line(lines[i].text, lineWidth, 9.0, 60.0 + i * 36.0, 0.0, 9.0, false, false);
			draw_menu_line(lines[teleportActiveLineIndex].text, lineWidth + 1.0, 11.0, 56.0 + teleportActiveLineIndex * 36.0, 0.0, 7.0, true, false);

			update_features();
			WAIT(0);
		} while (GetTickCount() < maxTickCount);
		waitTime = 0;

		// process buttons
		bool bSelect, bBack, bUp, bDown;
		get_button_state(&bSelect, &bBack, &bUp, &bDown, NULL, NULL);
		if (bSelect)
		{
			menu_beep();

			// get entity to teleport
			Entity e = PLAYER::PLAYER_PED_ID();
			if (PED::IS_PED_IN_ANY_VEHICLE(e, 0)) 
				e = PED::GET_VEHICLE_PED_IS_USING(e);

			// get coords
			Vector3 coords;
			bool success = false;
			if (teleportActiveLineIndex == 0) // marker
			{			
				bool blipFound = false;
				// search for marker blip
				int blipIterator = UI::_GET_BLIP_INFO_ID_ITERATOR();
				for (Blip i = UI::GET_FIRST_BLIP_INFO_ID(blipIterator); UI::DOES_BLIP_EXIST(i) != 0; i = UI::GET_NEXT_BLIP_INFO_ID(blipIterator))
				{
					if (UI::GET_BLIP_INFO_ID_TYPE(i) == 4) 
					{
						coords = UI::GET_BLIP_INFO_ID_COORD(i);
						blipFound = true;
						break;
					}
				}	
				if (blipFound)
				{
					// load needed map region and check height levels for ground existence
					bool groundFound = false;
					static float groundCheckHeight[] = {
						100.0, 150.0, 50.0, 0.0, 200.0, 250.0, 300.0, 350.0, 400.0, 
						450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0
					};					
					for (int i = 0; i < sizeof(groundCheckHeight) / sizeof(float); i++)
					{
						ENTITY::SET_ENTITY_COORDS_NO_OFFSET(e, coords.x, coords.y, groundCheckHeight[i], 0, 0, 1);
						WAIT(100);
						if (GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(coords.x, coords.y, groundCheckHeight[i], &coords.z, FALSE))
						{
							groundFound = true;
							coords.z += 3.0;
							break;
						}
					}
					// if ground not found then set Z in air and give player a parachute
					if (!groundFound)
					{
						coords.z = 1000.0;
						WEAPON::GIVE_DELAYED_WEAPON_TO_PED(PLAYER::PLAYER_PED_ID(), 0xFBAB5776, 1, 0);
					}
					success = true;
				} else
				{
					set_status_text("map marker isn't set");
				}

			} else // predefined coords
			{
				coords.x = lines[teleportActiveLineIndex].x;
				coords.y = lines[teleportActiveLineIndex].y;
				coords.z = lines[teleportActiveLineIndex].z;
				success = true;
			}

			// set player pos
			if (success)
			{
				ENTITY::SET_ENTITY_COORDS_NO_OFFSET(e, coords.x, coords.y, coords.z, 0, 0, 1);
				WAIT(0);
				set_status_text("teleported");
				return true;
			}
			
			waitTime = 200;
		} else
		if (bBack || trainer_switch_pressed())
		{
			menu_beep();
			break;
		} else
		if (bUp)
		{
			menu_beep();
			if (teleportActiveLineIndex == 0) 
				teleportActiveLineIndex = lineCount;
			teleportActiveLineIndex--;
			waitTime = 150;
		} else
		if (bDown)
		{
			menu_beep();
			teleportActiveLineIndex++;
			if (teleportActiveLineIndex == lineCount) 
				teleportActiveLineIndex = 0;			
			waitTime = 150;
		}
	}
	return false;
}

std::string line_as_str(std::string text, bool *pState)
{
	while (text.size() < 18) text += " ";
	return text + (pState ? (*pState ? " [ON]" : " [OFF]") : "");
}

int activeLineIndexPlayer = 0;

void process_player_menu()
{
	const float lineWidth = 250.0;
	const int lineCount = 15;
	
	std::string caption = "PLAYER  OPTIONS";

	static struct {
		LPCSTR		text;
		bool		*pState;
		bool		*pUpdated;
	} lines[lineCount] = {
		{"SKIN CHANGER", NULL, NULL},
		{"TELEPORT", NULL, NULL},
		{"FIX PLAYER", NULL, NULL},
		{"RESET SKIN", NULL, NULL},
		{"ADD CASH", NULL, NULL},
		{"WANTED UP", NULL, NULL},
		{"WANTED DOWN", NULL, NULL},
		{"NEVER WANTED", &featurePlayerNeverWanted, NULL},
		{"INVINCIBLE", &featurePlayerInvincible, &featurePlayerInvincibleUpdated},
		{"POLICE IGNORED", &featurePlayerIgnored, &featurePlayerIgnoredUpdated},
		{"UNLIM ABILITY", &featurePlayerUnlimitedAbility, NULL},
		{"NOISELESS", &featurePlayerNoNoise, &featurePlayerNoNoiseUpdated},
		{"FAST SWIM", &featurePlayerFastSwim, &featurePlayerFastSwimUpdated},
		{"FAST RUN", &featurePlayerFastRun, &featurePlayerFastRunUpdated},
		{"SUPER JUMP", &featurePlayerSuperJump, NULL}
	};

	DWORD waitTime = 150;
	while (true)
	{
		// timed menu draw, used for pause after active line switch
		DWORD maxTickCount = GetTickCount() + waitTime;
		do 
		{
			// draw menu
			draw_menu_line(caption, lineWidth, 15.0, 18.0, 0.0, 5.0, false, true);
			for (int i = 0; i < lineCount; i++)
				if (i != activeLineIndexPlayer)
					draw_menu_line(line_as_str(lines[i].text, lines[i].pState), 
								   lineWidth, 9.0, 60.0 + i * 36.0, 0.0, 9.0, false, false);
			draw_menu_line(line_as_str(lines[activeLineIndexPlayer].text, lines[activeLineIndexPlayer].pState), 
						   lineWidth + 1.0, 11.0, 56.0 + activeLineIndexPlayer * 36.0, 0.0, 7.0, true, false);

			update_features();
			WAIT(0);
		} while (GetTickCount() < maxTickCount);
		waitTime = 0;

		// process buttons
		bool bSelect, bBack, bUp, bDown;
		get_button_state(&bSelect, &bBack, &bUp, &bDown, NULL, NULL);
		if (bSelect)
		{
			menu_beep();

			// common variables
			BOOL bPlayerExists = ENTITY::DOES_ENTITY_EXIST(PLAYER::PLAYER_PED_ID());
			Player player = PLAYER::PLAYER_ID();
			Ped playerPed = PLAYER::PLAYER_PED_ID();

			switch (activeLineIndexPlayer)
			{
				// skin changer
				case 0:
					if (process_skinchanger_menu())	return;
					break;
				// teleport
				case 1:
					if (process_teleport_menu()) return;
					break;
				// fix player
				case 2:
					{
						ENTITY::SET_ENTITY_HEALTH(playerPed, ENTITY::GET_ENTITY_MAX_HEALTH(playerPed));
						PED::ADD_ARMOUR_TO_PED(playerPed, PLAYER::GET_PLAYER_MAX_ARMOUR(player) - PED::GET_PED_ARMOUR(playerPed));
						if (PED::IS_PED_IN_ANY_VEHICLE(playerPed, 0))
						{
							Vehicle playerVeh = PED::GET_VEHICLE_PED_IS_USING(playerPed);
							if (ENTITY::DOES_ENTITY_EXIST(playerVeh) && !ENTITY::IS_ENTITY_DEAD(playerVeh))
								VEHICLE::SET_VEHICLE_FIXED(playerVeh);
						}
						set_status_text("player fixed");
					}
					break;
				// reset model skin
				case 3:
					{
						PED::SET_PED_DEFAULT_COMPONENT_VARIATION(playerPed);
						set_status_text("using default model skin");
					}
					break;
				// add cash
				case 4: 
					for (int i = 0; i < 3; i++)
					{
						char statNameFull[32];
						sprintf_s(statNameFull, "SP%d_TOTAL_CASH", i);
						Hash hash = GAMEPLAY::GET_HASH_KEY(statNameFull);
						int val;
						STATS::STAT_GET_INT(hash, &val, -1);
						val += 1000000;
						STATS::STAT_SET_INT(hash, val, 1);
					}
					set_status_text("cash added");
					break;
				// wanted up
				case 5:	
					if (bPlayerExists && PLAYER::GET_PLAYER_WANTED_LEVEL(player) < 5)
					{
						PLAYER::SET_PLAYER_WANTED_LEVEL(player, PLAYER::GET_PLAYER_WANTED_LEVEL(player) + 1, 0);
						PLAYER::SET_PLAYER_WANTED_LEVEL_NOW(player, 0);
						set_status_text("wanted up");
					}
					break;
				// wanted down
				case 6:	
					if (bPlayerExists && PLAYER::GET_PLAYER_WANTED_LEVEL(player) > 0)
					{
						PLAYER::SET_PLAYER_WANTED_LEVEL(player, PLAYER::GET_PLAYER_WANTED_LEVEL(player) - 1, 0);
						PLAYER::SET_PLAYER_WANTED_LEVEL_NOW(player, 0);
						set_status_text("wanted down");
					}
					break;
				// switchable features
				default:
					if (lines[activeLineIndexPlayer].pState)
						*lines[activeLineIndexPlayer].pState = !(*lines[activeLineIndexPlayer].pState);
					if (lines[activeLineIndexPlayer].pUpdated)
						*lines[activeLineIndexPlayer].pUpdated = true;					
			}
			waitTime = 200;
		} else
		if (bBack || trainer_switch_pressed())
		{
			menu_beep();
			break;
		} else
		if (bUp)
		{
			menu_beep();
			if (activeLineIndexPlayer == 0) 
				activeLineIndexPlayer = lineCount;
			activeLineIndexPlayer--;
			waitTime = 150;
		} else
		if (bDown)
		{
			menu_beep();
			activeLineIndexPlayer++;
			if (activeLineIndexPlayer == lineCount) 
				activeLineIndexPlayer = 0;			
			waitTime = 150;
		}
	}
}

int activeLineIndexWeapon = 0;

void process_weapon_menu()
{
	const float lineWidth = 250.0;
	const int lineCount = 6;

	std::string caption = "WEAPON  OPTIONS";

	static struct {
		LPCSTR		text;
		bool		*pState;
		bool		*pUpdated;
	} lines[lineCount] = {
		{"GET ALL WEAPON",	NULL,						  NULL},
		{"NO RELOAD",		&featureWeaponNoReload,		  NULL},
		{"FIRE AMMO",		&featureWeaponFireAmmo,		  NULL},
		{"EXPLOSIVE AMMO",  &featureWeaponExplosiveAmmo,  NULL},
		{"EXPLOSIVE MELEE", &featureWeaponExplosiveMelee, NULL},
		{"VEHICLE ROCKETS", &featureWeaponVehRockets,	  NULL}
	};

	static LPCSTR weaponNames[] = {
		"WEAPON_KNIFE", "WEAPON_NIGHTSTICK", "WEAPON_HAMMER", "WEAPON_BAT", "WEAPON_GOLFCLUB", "WEAPON_CROWBAR", 
		"WEAPON_PISTOL", "WEAPON_COMBATPISTOL", "WEAPON_APPISTOL", "WEAPON_PISTOL50", "WEAPON_MICROSMG", "WEAPON_SMG", 
		"WEAPON_ASSAULTSMG", "WEAPON_ASSAULTRIFLE", "WEAPON_CARBINERIFLE", "WEAPON_ADVANCEDRIFLE", "WEAPON_MG",
		"WEAPON_COMBATMG", "WEAPON_PUMPSHOTGUN", "WEAPON_SAWNOFFSHOTGUN", "WEAPON_ASSAULTSHOTGUN", "WEAPON_BULLPUPSHOTGUN",
		"WEAPON_STUNGUN", "WEAPON_SNIPERRIFLE", "WEAPON_HEAVYSNIPER", "WEAPON_GRENADELAUNCHER", "WEAPON_GRENADELAUNCHER_SMOKE",
		"WEAPON_RPG", "WEAPON_MINIGUN", "WEAPON_GRENADE", "WEAPON_STICKYBOMB", "WEAPON_SMOKEGRENADE", "WEAPON_BZGAS",
		"WEAPON_MOLOTOV", "WEAPON_FIREEXTINGUISHER", "WEAPON_PETROLCAN",
		"WEAPON_SNSPISTOL", "WEAPON_SPECIALCARBINE", "WEAPON_HEAVYPISTOL", "WEAPON_BULLPUPRIFLE", "WEAPON_HOMINGLAUNCHER",
		"WEAPON_PROXMINE", "WEAPON_SNOWBALL", "WEAPON_VINTAGEPISTOL", "WEAPON_DAGGER", "WEAPON_FIREWORK", "WEAPON_MUSKET",
		"WEAPON_MARKSMANRIFLE", "WEAPON_HEAVYSHOTGUN", "WEAPON_GUSENBERG", "WEAPON_HATCHET", "WEAPON_RAILGUN",
		"WEAPON_COMBATPDW", "WEAPON_KNUCKLE", "WEAPON_MARKSMANPISTOL",
		"WEAPON_FLASHLIGHT", "WEAPON_MACHETE", "WEAPON_MACHINEPISTOL",
		"WEAPON_SWITCHBLADE", "WEAPON_REVOLVER"
	};

	DWORD waitTime = 150;
	while (true)
	{
		// timed menu draw, used for pause after active line switch
		DWORD maxTickCount = GetTickCount() + waitTime;
		do 
		{
			// draw menu
			draw_menu_line(caption, lineWidth, 15.0, 18.0, 0.0, 5.0, false, true);
			for (int i = 0; i < lineCount; i++)
				if (i != activeLineIndexWeapon)
					draw_menu_line(line_as_str(lines[i].text, lines[i].pState), 
								   lineWidth, 9.0, 60.0 + i * 36.0, 0.0, 9.0, false, false);
			draw_menu_line(line_as_str(lines[activeLineIndexWeapon].text, lines[activeLineIndexWeapon].pState), 
						   lineWidth + 1.0, 11.0, 56.0 + activeLineIndexWeapon * 36.0, 0.0, 7.0, true, false);

			update_features();
			WAIT(0);
		} while (GetTickCount() < maxTickCount);
		waitTime = 0;

		// process buttons
		bool bSelect, bBack, bUp, bDown;
		get_button_state(&bSelect, &bBack, &bUp, &bDown, NULL, NULL);
		if (bSelect)
		{
			menu_beep();

			// common variables
			BOOL bPlayerExists = ENTITY::DOES_ENTITY_EXIST(PLAYER::PLAYER_PED_ID());
			Player player = PLAYER::PLAYER_ID();
			Ped playerPed = PLAYER::PLAYER_PED_ID();

			switch (activeLineIndexWeapon)
			{
				case 0:
					for (int i = 0; i < sizeof(weaponNames) / sizeof(weaponNames[0]); i++)
						WEAPON::GIVE_DELAYED_WEAPON_TO_PED(playerPed, GAMEPLAY::GET_HASH_KEY((char *)weaponNames[i]), 1000, 0);
					set_status_text("all weapon added");
					break;
				// switchable features
				default:
					if (lines[activeLineIndexWeapon].pState)
						*lines[activeLineIndexWeapon].pState = !(*lines[activeLineIndexWeapon].pState);
					if (lines[activeLineIndexWeapon].pUpdated)
						*lines[activeLineIndexWeapon].pUpdated = true;					
			}
			waitTime = 200;
		} else
		if (bBack || trainer_switch_pressed())
		{
			menu_beep();
			break;
		} else
		if (bUp)
		{
			menu_beep();
			if (activeLineIndexWeapon == 0) 
				activeLineIndexWeapon = lineCount;
			activeLineIndexWeapon--;
			waitTime = 150;
		} else
		if (bDown)
		{
			menu_beep();
			activeLineIndexWeapon++;
			if (activeLineIndexWeapon == lineCount) 
				activeLineIndexWeapon = 0;			
			waitTime = 150;
		}
	}
}

LPCSTR vehicleModels[40][10] = {
	{"NINEF", "NINEF2", "BLISTA", "ASEA", "ASEA2", "BOATTRAILER", "BUS", "ARMYTANKER", "ARMYTRAILER", "ARMYTRAILER2"},
	{"SUNTRAP", "COACH", "AIRBUS", "ASTEROPE", "AIRTUG", "AMBULANCE", "BARRACKS", "BARRACKS2", "BALLER", "BALLER2"},
	{"BJXL", "BANSHEE", "BENSON", "BFINJECTION", "BIFF", "BLAZER", "BLAZER2", "BLAZER3", "BISON", "BISON2"},
	{"BISON3", "BOXVILLE", "BOXVILLE2", "BOXVILLE3", "BOBCATXL", "BODHI2", "BUCCANEER", "BUFFALO", "BUFFALO2", "BULLDOZER"},
	{"BULLET", "BLIMP", "BURRITO", "BURRITO2", "BURRITO3", "BURRITO4", "BURRITO5", "CAVALCADE", "CAVALCADE2", "POLICET"},
	{"GBURRITO", "CABLECAR", "CADDY", "CADDY2", "CAMPER", "CARBONIZZARE", "CHEETAH", "COMET2", "COGCABRIO", "COQUETTE"},
	{"CUTTER", "GRESLEY", "DILETTANTE", "DILETTANTE2", "DUNE", "DUNE2", "HOTKNIFE", "DLOADER", "DUBSTA", "DUBSTA2"},
	{"DUMP", "RUBBLE", "DOCKTUG", "DOMINATOR", "EMPEROR", "EMPEROR2", "EMPEROR3", "ENTITYXF", "EXEMPLAR", "ELEGY2"},
	{"F620", "FBI", "FBI2", "FELON", "FELON2", "FELTZER2", "FIRETRUK", "FLATBED", "FORKLIFT", "FQ2"},
	{"FUSILADE", "FUGITIVE", "FUTO", "GRANGER", "GAUNTLET", "HABANERO", "HAULER", "HANDLER", "INFERNUS", "INGOT"},
	{"INTRUDER", "ISSI2", "JACKAL", "JOURNEY", "JB700", "KHAMELION", "LANDSTALKER", "LGUARD", "MANANA", "MESA"},
	{"MESA2", "MESA3", "CRUSADER", "MINIVAN", "MIXER", "MIXER2", "MONROE", "MOWER", "MULE", "MULE2"},
	{"ORACLE", "ORACLE2", "PACKER", "PATRIOT", "PBUS", "PENUMBRA", "PEYOTE", "PHANTOM", "PHOENIX", "PICADOR"},
	{"POUNDER", "POLICE", "POLICE4", "POLICE2", "POLICE3", "POLICEOLD1", "POLICEOLD2", "PONY", "PONY2", "PRAIRIE"},
	{"PRANGER", "PREMIER", "PRIMO", "PROPTRAILER", "RANCHERXL", "RANCHERXL2", "RAPIDGT", "RAPIDGT2", "RADI", "RATLOADER"},
	{"REBEL", "REGINA", "REBEL2", "RENTALBUS", "RUINER", "RUMPO", "RUMPO2", "RHINO", "RIOT", "RIPLEY"},
	{"ROCOTO", "ROMERO", "SABREGT", "SADLER", "SADLER2", "SANDKING", "SANDKING2", "SCHAFTER2", "SCHWARZER", "SCRAP"},
	{"SEMINOLE", "SENTINEL", "SENTINEL2", "ZION", "ZION2", "SERRANO", "SHERIFF", "SHERIFF2", "SPEEDO", "SPEEDO2"},
	{"STANIER", "STINGER", "STINGERGT", "STOCKADE", "STOCKADE3", "STRATUM", "SULTAN", "SUPERD", "SURANO", "SURFER"},
	{"SURFER2", "SURGE", "TACO", "TAILGATER", "TAXI", "TRASH", "TRACTOR", "TRACTOR2", "TRACTOR3", "GRAINTRAILER"},
	{"BALETRAILER", "TIPTRUCK", "TIPTRUCK2", "TORNADO", "TORNADO2", "TORNADO3", "TORNADO4", "TOURBUS", "TOWTRUCK", "TOWTRUCK2"},
	{"UTILLITRUCK", "UTILLITRUCK2", "UTILLITRUCK3", "VOODOO2", "WASHINGTON", "STRETCH", "YOUGA", "ZTYPE", "SANCHEZ", "SANCHEZ2"},
	{"SCORCHER", "TRIBIKE", "TRIBIKE2", "TRIBIKE3", "FIXTER", "CRUISER", "BMX", "POLICEB", "AKUMA", "CARBONRS"},
	{"BAGGER", "BATI", "BATI2", "RUFFIAN", "DAEMON", "DOUBLE", "PCJ", "VADER", "VIGERO", "FAGGIO2"},
	{"HEXER", "ANNIHILATOR", "BUZZARD", "BUZZARD2", "CARGOBOB", "CARGOBOB2", "CARGOBOB3", "SKYLIFT", "POLMAV", "MAVERICK"},
	{"NEMESIS", "FROGGER", "FROGGER2", "CUBAN800", "DUSTER", "STUNT", "MAMMATUS", "JET", "SHAMAL", "LUXOR"},
	{"TITAN", "LAZER", "CARGOPLANE", "SQUALO", "MARQUIS", "DINGHY", "DINGHY2", "JETMAX", "PREDATOR", "TROPIC"},
	{"SEASHARK", "SEASHARK2", "SUBMERSIBLE", "TRAILERS", "TRAILERS2", "TRAILERS3", "TVTRAILER", "RAKETRAILER", "TANKER", "TRAILERLOGS"},
	{"TR2", "TR3", "TR4", "TRFLAT", "TRAILERSMALL", "VELUM", "ADDER", "VOLTIC", "VACCA", "BIFTA"},
	{ "SPEEDER", "PARADISE", "KALAHARI", "JESTER", "TURISMOR", "VESTRA", "ALPHA", "HUNTLEY", "THRUST", "MASSACRO" },
	{ "MASSACRO2", "ZENTORNO", "BLADE", "GLENDALE", "PANTO", "PIGALLE", "WARRENER", "RHAPSODY", "DUBSTA3", "MONSTER" },
	{ "SOVEREIGN", "INNOVATION", "HAKUCHOU", "FUROREGT", "MILJET", "COQUETTE2", "BTYPE", "BUFFALO3", "DOMINATOR2", "GAUNTLET2" },
	{ "MARSHALL", "DUKES", "DUKES2", "STALION", "STALION2", "BLISTA2", "BLISTA3", "DODO", "SUBMERSIBLE2", "HYDRA" },
	{ "INSURGENT", "INSURGENT2", "TECHNICAL", "SAVAGE", "VALKYRIE", "KURUMA", "KURUMA2", "JESTER2", "CASCO", "VELUM2" },
	{ "GUARDIAN", "ENDURO", "LECTRO", "SLAMVAN", "SLAMVAN2", "RATLOADER2", "SWIFT2", "LUXOR2", "FELTZER3", "OSIRIS" },
	{ "VIRGO", "WINDSOR", "BESRA", "SWIFT", "BLIMP2", "VINDICATOR", "TORO", "T20", "COQUETTE3", "CHINO" },
	{ "BRAWLER", "BUCCANEER2", "CHINO2", "FACTION", "FACTION2", "MOONBEAM", "MOONBEAM2", "PRIMO2", "VOODOO", "LURCHER" },
	{ "BTYPE2", "BALLER3", "BALLER4", "BALLER5", "BALLER6", "CARGOBOB4", "COG55", "COG552", "COGNOSCENTI", "COGNOSCENTI2" },
	{ "DINGHY4", "LIMO2", "MAMBA", "NIGHTSHADE", "SCHAFTER3", "SCHAFTER4", "SCHAFTER5", "SCHAFTER6", "SEASHARK3", "SPEEDER2" },
	{ "SUPERVOLITO", "SUPERVOLITO2", "TORO2", "TROPIC2", "VALKYRIE2", "VERLIERER2", "TAMPA", "BANSHEE2", "SULTANRS", "BTYPE3" }
};

int carspawnActiveLineIndex = 0;
int carspawnActiveItemIndex = 0;

bool process_carspawn_menu()
{
	DWORD waitTime = 150;
	const int lineCount = 40;
	const int itemCount = 10;
	const int itemCountLastLine = 10;
	while (true)
	{
		// timed menu draw, used for pause after active line switch
		DWORD maxTickCount = GetTickCount() + waitTime;
		do 
		{
			// draw menu
			char caption[32];
			sprintf_s(caption, "CAR SPAWNER   %d / %d", carspawnActiveLineIndex + 1, lineCount);
			draw_menu_line(caption, 350.0, 15.0, 18.0, 0.0, 5.0, false, true);
			for (int i = 0; i < itemCount; i++)
				if (strlen(vehicleModels[carspawnActiveLineIndex][i]))
					draw_menu_line(vehicleModels[carspawnActiveLineIndex][i], 100.0, 5.0, 200.0, 100.0 + i * 110.0, 5.0, i == carspawnActiveItemIndex, false, false);
			
			update_features();
			WAIT(0);
		} while (GetTickCount() < maxTickCount);
		waitTime = 0;

		bool bSelect, bBack, bUp, bDown, bLeft, bRight;
		get_button_state(&bSelect, &bBack, &bUp, &bDown, &bLeft, &bRight);
		
		if (bSelect)
		{
			menu_beep();
			LPCSTR modelName = vehicleModels[carspawnActiveLineIndex][carspawnActiveItemIndex];
			DWORD model = GAMEPLAY::GET_HASH_KEY((char *)modelName);
			if (STREAMING::IS_MODEL_IN_CDIMAGE(model) && STREAMING::IS_MODEL_A_VEHICLE(model))
			{
				STREAMING::REQUEST_MODEL(model);				
				while (!STREAMING::HAS_MODEL_LOADED(model)) WAIT(0);
				Vector3 coords = ENTITY::GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(PLAYER::PLAYER_PED_ID(), 0.0, 5.0, 0.0);
				Vehicle veh = VEHICLE::CREATE_VEHICLE(model, coords.x, coords.y, coords.z, 0.0, 1, 1);
				VEHICLE::SET_VEHICLE_ON_GROUND_PROPERLY(veh);

				if (featureVehWrapInSpawned)
				{
					ENTITY::SET_ENTITY_HEADING(veh, ENTITY::GET_ENTITY_HEADING(PLAYER::PLAYER_PED_ID()));
					PED::SET_PED_INTO_VEHICLE(PLAYER::PLAYER_PED_ID(), veh, -1);
				}

				WAIT(0);
				STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(model);
				ENTITY::SET_VEHICLE_AS_NO_LONGER_NEEDED(&veh);

				char statusText[32];
				sprintf_s(statusText, "%s spawned", modelName);
				set_status_text(statusText);

				return true;
			}
		} else
		if (bBack)
		{
			menu_beep();
			break;
		} else
		if (bRight)
		{
			menu_beep();
			carspawnActiveItemIndex++;
			int itemsMax = (carspawnActiveLineIndex == (lineCount - 1)) ? itemCountLastLine : itemCount;
			if (carspawnActiveItemIndex == itemsMax) 
				carspawnActiveItemIndex = 0;			
			waitTime = 100;
		} else
		if (bLeft)
		{
			menu_beep();
			if (carspawnActiveItemIndex == 0) 
				carspawnActiveItemIndex = (carspawnActiveLineIndex == (lineCount - 1)) ? itemCountLastLine : itemCount;
			carspawnActiveItemIndex--;
			waitTime = 100;
		} else
		if (bUp)
		{
			menu_beep();
			if (carspawnActiveLineIndex == 0) 
				carspawnActiveLineIndex = lineCount;
			carspawnActiveLineIndex--;
			waitTime = 200;
		} else
		if (bDown)
		{
			menu_beep();
			carspawnActiveLineIndex++;
			if (carspawnActiveLineIndex == lineCount) 
				carspawnActiveLineIndex = 0;			
			waitTime = 200;
		}
		if (carspawnActiveLineIndex == (lineCount - 1))
			if (carspawnActiveItemIndex >= itemCountLastLine)
				carspawnActiveItemIndex = 0;
	}
	return false;
}

int activeLineIndexVeh = 0;

void process_veh_menu()
{
	const float lineWidth = 250.0;
	const int lineCount = 8;

	std::string caption = "VEHICLE  OPTIONS";

	static struct {
		LPCSTR		text;
		bool		*pState;
		bool		*pUpdated;
	} lines[lineCount] = {
		{"CAR SPAWNER",		NULL, NULL},
		{"PAINT RANDOM",	NULL, NULL},
		{"FIX",				NULL, NULL},
		{"SEAT BELT",		&featureVehSeatbelt, &featureVehSeatbeltUpdated},
		{"WRAP IN SPAWNED",	&featureVehWrapInSpawned, NULL},
		{"INVINCIBLE",		&featureVehInvincible, &featureVehInvincibleUpdated},
		{"STRONG WHEELS",	&featureVehInvincibleWheels, &featureVehInvincibleWheelsUpdated},
		{"SPEED BOOST",		&featureVehSpeedBoost, NULL}		
	};

	DWORD waitTime = 150;
	while (true)
	{
		// timed menu draw, used for pause after active line switch
		DWORD maxTickCount = GetTickCount() + waitTime;
		do 
		{
			// draw menu
			draw_menu_line(caption, lineWidth, 15.0, 18.0, 0.0, 5.0, false, true);
			for (int i = 0; i < lineCount; i++)
				if (i != activeLineIndexVeh)
					draw_menu_line(line_as_str(lines[i].text, lines[i].pState), 
								   lineWidth, 9.0, 60.0 + i * 36.0, 0.0, 9.0, false, false);
			draw_menu_line(line_as_str(lines[activeLineIndexVeh].text, lines[activeLineIndexVeh].pState), 
						   lineWidth + 1.0, 11.0, 56.0 + activeLineIndexVeh * 36.0, 0.0, 7.0, true, false);

			update_features();
			WAIT(0);
		} while (GetTickCount() < maxTickCount);
		waitTime = 0;

		// process buttons
		bool bSelect, bBack, bUp, bDown;
		get_button_state(&bSelect, &bBack, &bUp, &bDown, NULL, NULL);
		if (bSelect)
		{
			menu_beep();

			// common variables
			BOOL bPlayerExists = ENTITY::DOES_ENTITY_EXIST(PLAYER::PLAYER_PED_ID());
			Player player = PLAYER::PLAYER_ID();
			Ped playerPed = PLAYER::PLAYER_PED_ID();

			switch (activeLineIndexVeh)
			{
				case 0:
					if (process_carspawn_menu()) return;				
					break;
				case 1:
					if (bPlayerExists) 
					{
						if (PED::IS_PED_IN_ANY_VEHICLE(playerPed, 0))
						{
							Vehicle veh = PED::GET_VEHICLE_PED_IS_USING(playerPed);
							VEHICLE::SET_VEHICLE_CUSTOM_PRIMARY_COLOUR(veh, rand() % 255, rand() % 255, rand() % 255);
							if (VEHICLE::GET_IS_VEHICLE_PRIMARY_COLOUR_CUSTOM(veh))
								VEHICLE::SET_VEHICLE_CUSTOM_SECONDARY_COLOUR(veh, rand() % 255, rand() % 255, rand() % 255);
						} else
						{
							set_status_text("player isn't in a vehicle");
						}
					}
					break;
				case 2:
					if (bPlayerExists) 
						if (PED::IS_PED_IN_ANY_VEHICLE(playerPed, 0))
							VEHICLE::SET_VEHICLE_FIXED(PED::GET_VEHICLE_PED_IS_USING(playerPed));
						else
							set_status_text("player isn't in a vehicle");
					break;
				// switchable features
				default:
					if (lines[activeLineIndexVeh].pState)
						*lines[activeLineIndexVeh].pState = !(*lines[activeLineIndexVeh].pState);
					if (lines[activeLineIndexVeh].pUpdated)
						*lines[activeLineIndexVeh].pUpdated = true;					
			}
			waitTime = 200;
		} else
		if (bBack || trainer_switch_pressed())
		{
			menu_beep();
			break;
		} else
		if (bUp)
		{
			menu_beep();
			if (activeLineIndexVeh == 0) 
				activeLineIndexVeh = lineCount;
			activeLineIndexVeh--;
			waitTime = 150;
		} else
		if (bDown)
		{
			menu_beep();
			activeLineIndexVeh++;
			if (activeLineIndexVeh == lineCount) 
				activeLineIndexVeh = 0;			
			waitTime = 150;
		}
	}
}

int activeLineIndexWorld = 0;

void process_world_menu()
{
	const float lineWidth = 250.0;
	const int lineCount = 5;

	std::string caption = "WORLD  OPTIONS";

	static struct {
		LPCSTR		text;
		bool		*pState;
		bool		*pUpdated;
	} lines[lineCount] = {
		{"MOON GRAVITY",	&featureWorldMoonGravity,	NULL},
		{"RANDOM COPS",		&featureWorldRandomCops,	NULL},
		{"RANDOM TRAINS",	&featureWorldRandomTrains,	NULL},
		{"RANDOM BOATS",	&featureWorldRandomBoats,	NULL},
		{"GARBAGE TRUCKS",	&featureWorldGarbageTrucks,	NULL}
	};

	DWORD waitTime = 150;
	while (true)
	{
		// timed menu draw, used for pause after active line switch
		DWORD maxTickCount = GetTickCount() + waitTime;
		do 
		{
			// draw menu
			draw_menu_line(caption, lineWidth, 15.0, 18.0, 0.0, 5.0, false, true);
			for (int i = 0; i < lineCount; i++)
				if (i != activeLineIndexWorld)
					draw_menu_line(line_as_str(lines[i].text, lines[i].pState), 
					lineWidth, 9.0, 60.0 + i * 36.0, 0.0, 9.0, false, false);
			draw_menu_line(line_as_str(lines[activeLineIndexWorld].text, lines[activeLineIndexWorld].pState), 
				lineWidth + 1.0, 11.0, 56.0 + activeLineIndexWorld * 36.0, 0.0, 7.0, true, false);

			update_features();
			WAIT(0);
		} while (GetTickCount() < maxTickCount);
		waitTime = 0;

		// process buttons
		bool bSelect, bBack, bUp, bDown;
		get_button_state(&bSelect, &bBack, &bUp, &bDown, NULL, NULL);
		if (bSelect)
		{
			menu_beep();
			switch (activeLineIndexWorld)
			{
				case 0: 
					featureWorldMoonGravity = !featureWorldMoonGravity;
					GAMEPLAY::SET_GRAVITY_LEVEL(featureWorldMoonGravity ? 2 : 0);
					break;
				case 1: 
					// featureWorldRandomCops being set in update_features
					PED::SET_CREATE_RANDOM_COPS(!featureWorldRandomCops);
					break;
				case 2:
					featureWorldRandomTrains = !featureWorldRandomTrains;
					VEHICLE::SET_RANDOM_TRAINS(featureWorldRandomTrains);
					break;
				case 3:
					featureWorldRandomBoats = !featureWorldRandomBoats;
					VEHICLE::SET_RANDOM_BOATS(featureWorldRandomBoats);
					break;
				case 4:
					featureWorldGarbageTrucks = !featureWorldGarbageTrucks;
					VEHICLE::SET_GARBAGE_TRUCKS(featureWorldGarbageTrucks);
					break;
			}
			waitTime = 200;
		} else
		if (bBack || trainer_switch_pressed())
		{
			menu_beep();
			break;
		} else
		if (bUp)
		{
			menu_beep();
			if (activeLineIndexWorld == 0) 
				activeLineIndexWorld = lineCount;
			activeLineIndexWorld--;
			waitTime = 150;
		} else
		if (bDown)
		{
			menu_beep();
			activeLineIndexWorld++;
			if (activeLineIndexWorld == lineCount) 
				activeLineIndexWorld = 0;			
			waitTime = 150;
		}
	}
}

int activeLineIndexTime = 0;

void process_time_menu()
{
	const float lineWidth = 250.0;
	const int lineCount = 4;

	std::string caption = "TIME  OPTIONS";

	static struct {
		LPCSTR		text;
		bool		*pState;
		bool		*pUpdated;
	} lines[lineCount] = {
		{"HOUR FORWARD",	 NULL,				 NULL},
		{"HOUR BACKWARD",	 NULL,				 NULL},
		{"CLOCK PAUSED",	 &featureTimePaused, &featureTimePausedUpdated},
		{"SYNC WITH SYSTEM", &featureTimeSynced, NULL}
	};

	DWORD waitTime = 150;
	while (true)
	{
		// timed menu draw, used for pause after active line switch
		DWORD maxTickCount = GetTickCount() + waitTime;
		do 
		{
			// draw menu
			draw_menu_line(caption, lineWidth, 15.0, 18.0, 0.0, 5.0, false, true);
			for (int i = 0; i < lineCount; i++)
				if (i != activeLineIndexTime)
					draw_menu_line(line_as_str(lines[i].text, lines[i].pState), 
					lineWidth, 9.0, 60.0 + i * 36.0, 0.0, 9.0, false, false);
			draw_menu_line(line_as_str(lines[activeLineIndexTime].text, lines[activeLineIndexTime].pState), 
				lineWidth + 1.0, 11.0, 56.0 + activeLineIndexTime * 36.0, 0.0, 7.0, true, false);

			update_features();
			WAIT(0);
		} while (GetTickCount() < maxTickCount);
		waitTime = 0;

		// process buttons
		bool bSelect, bBack, bUp, bDown;
		get_button_state(&bSelect, &bBack, &bUp, &bDown, NULL, NULL);
		if (bSelect)
		{
			menu_beep();
			switch (activeLineIndexTime)
			{
			// hour forward/backward
			case 0: 
			case 1:
				{
					int h = TIME::GET_CLOCK_HOURS();
					if (activeLineIndexTime == 0) h = (h == 23) ? 0 : h + 1; else h = (h == 0) ? 23 : h - 1;
					int m = TIME::GET_CLOCK_MINUTES();
					TIME::SET_CLOCK_TIME(h, m, 0);
					char text[32];
					sprintf_s(text, "time %d:%d", h, m);
					set_status_text(text);
				}
				break;
			// switchable features
			default:
				if (lines[activeLineIndexTime].pState)
					*lines[activeLineIndexTime].pState = !(*lines[activeLineIndexTime].pState);
				if (lines[activeLineIndexTime].pUpdated)
					*lines[activeLineIndexTime].pUpdated = true;	
			}
			waitTime = 200;
		} else
		if (bBack || trainer_switch_pressed())
		{
			menu_beep();
			break;
		} else
		if (bUp)
		{
			menu_beep();
			if (activeLineIndexTime == 0) 
				activeLineIndexTime = lineCount;
			activeLineIndexTime--;
			waitTime = 150;
		} else
		if (bDown)
		{
			menu_beep();
			activeLineIndexTime++;
			if (activeLineIndexTime == lineCount) 
				activeLineIndexTime = 0;			
			waitTime = 150;
		}
	}
}

int activeLineIndexWeather = 0;

void process_weather_menu()
{
	const float lineWidth = 250.0;
	const int lineCount = 16;

	std::string caption = "WEATHER  OPTIONS";

	static struct {
		LPCSTR		text;
		bool		*pState;
		bool		*pUpdated;
	} lines[lineCount] = {
		{"WIND",	 	 &featureWeatherWind,	NULL},
		{"SET PERSIST",  &featureWeatherPers,	NULL},
		{"EXTRASUNNY",	 NULL,					NULL},
		{"CLEAR",		 NULL,					NULL},
		{"CLOUDS",		 NULL,					NULL},
		{"SMOG",		 NULL,					NULL},
		{"FOGGY",	 	 NULL,					NULL},
		{"OVERCAST",	 NULL,					NULL},
		{"RAIN",		 NULL,					NULL},
		{"THUNDER",		 NULL,					NULL},
		{"CLEARING",	 NULL,					NULL},
		{"NEUTRAL",		 NULL,					NULL},
		{"SNOW",		 NULL,					NULL},
		{"BLIZZARD",	 NULL,					NULL},
		{"SNOWLIGHT",	 NULL,					NULL},
		{"XMAS",		 NULL,					NULL}
	};


	DWORD waitTime = 150;
	while (true)
	{
		// timed menu draw, used for pause after active line switch
		DWORD maxTickCount = GetTickCount() + waitTime;
		do 
		{
			// draw menu
			draw_menu_line(caption, lineWidth, 15.0, 18.0, 0.0, 5.0, false, true);
			for (int i = 0; i < lineCount; i++)
				if (i != activeLineIndexWeather)
					draw_menu_line(line_as_str(lines[i].text, lines[i].pState), 
					lineWidth, 9.0, 60.0 + i * 36.0, 0.0, 9.0, false, false);
			draw_menu_line(line_as_str(lines[activeLineIndexWeather].text, lines[activeLineIndexWeather].pState), 
				lineWidth + 1.0, 11.0, 56.0 + activeLineIndexWeather * 36.0, 0.0, 7.0, true, false);

			update_features();
			WAIT(0);
		} while (GetTickCount() < maxTickCount);
		waitTime = 0;

		// process buttons
		bool bSelect, bBack, bUp, bDown;
		get_button_state(&bSelect, &bBack, &bUp, &bDown, NULL, NULL);
		if (bSelect)
		{
			menu_beep();
			switch (activeLineIndexWeather)
			{
			// wind
			case 0: 
				featureWeatherWind = !featureWeatherWind;
				if (featureWeatherWind)
				{
					GAMEPLAY::SET_WIND(1.0);
					GAMEPLAY::SET_WIND_SPEED(11.99);
					GAMEPLAY::SET_WIND_DIRECTION(ENTITY::GET_ENTITY_HEADING(PLAYER::PLAYER_PED_ID()));
				} else
				{
					GAMEPLAY::SET_WIND(0.0);
					GAMEPLAY::SET_WIND_SPEED(0.0);
				}
				break;
			// set persist
			case 1:
				featureWeatherPers = !featureWeatherPers;
				break;
			// set weather
			default:
				GAMEPLAY::CLEAR_OVERRIDE_WEATHER();
				if (featureWeatherPers)
				{
					GAMEPLAY::SET_OVERRIDE_WEATHER((char *)lines[activeLineIndexWeather].text);
				}
				else
				{
					GAMEPLAY::SET_WEATHER_TYPE_NOW_PERSIST((char *)lines[activeLineIndexWeather].text);
					GAMEPLAY::CLEAR_WEATHER_TYPE_PERSIST();
				}				
				set_status_text(lines[activeLineIndexWeather].text);
			}
			waitTime = 200;
		} else
		if (bBack || trainer_switch_pressed())
		{
			menu_beep();
			break;
		} else
		if (bUp)
		{
			menu_beep();
			if (activeLineIndexWeather == 0) 
				activeLineIndexWeather = lineCount;
			activeLineIndexWeather--;
			waitTime = 150;
		} else
		if (bDown)
		{
			menu_beep();
			activeLineIndexWeather++;
			if (activeLineIndexWeather == lineCount) 
				activeLineIndexWeather = 0;			
			waitTime = 150;
		}
	}
}

int activeLineIndexMisc = 0;

void process_misc_menu()
{
	const float lineWidth = 250.0;
	const int lineCount = 2;

	std::string caption = "MISC  OPTIONS";

	static struct {
		LPCSTR		text;
		bool		*pState;
		bool		*pUpdated;
	} lines[lineCount] = {
		{"NEXT RADIO TRACK",	NULL,					NULL},
		{"HIDE HUD",			&featureMiscHideHud,	NULL}		
	};


	DWORD waitTime = 150;
	while (true)
	{
		// timed menu draw, used for pause after active line switch
		DWORD maxTickCount = GetTickCount() + waitTime;
		do 
		{
			// draw menu
			draw_menu_line(caption, lineWidth, 15.0, 18.0, 0.0, 5.0, false, true);
			for (int i = 0; i < lineCount; i++)
				if (i != activeLineIndexMisc)
					draw_menu_line(line_as_str(lines[i].text, lines[i].pState), 
					lineWidth, 9.0, 60.0 + i * 36.0, 0.0, 9.0, false, false);
			draw_menu_line(line_as_str(lines[activeLineIndexMisc].text, lines[activeLineIndexMisc].pState), 
				lineWidth + 1.0, 11.0, 56.0 + activeLineIndexMisc * 36.0, 0.0, 7.0, true, false);

			update_features();
			WAIT(0);
		} while (GetTickCount() < maxTickCount);
		waitTime = 0;

		// process buttons
		bool bSelect, bBack, bUp, bDown;
		get_button_state(&bSelect, &bBack, &bUp, &bDown, NULL, NULL);
		if (bSelect)
		{
			menu_beep();
			switch (activeLineIndexMisc)
			{
			// next radio track
			case 0: 
				if (ENTITY::DOES_ENTITY_EXIST(PLAYER::PLAYER_PED_ID()) && 
					PED::IS_PED_IN_ANY_VEHICLE(PLAYER::PLAYER_PED_ID(), 0))
						AUDIO::SKIP_RADIO_FORWARD();
				break;
			// switchable features
			default:
				if (lines[activeLineIndexMisc].pState)
					*lines[activeLineIndexMisc].pState = !(*lines[activeLineIndexMisc].pState);
				if (lines[activeLineIndexMisc].pUpdated)
					*lines[activeLineIndexMisc].pUpdated = true;
			}
			waitTime = 200;
		} else
		if (bBack || trainer_switch_pressed())
		{
			menu_beep();
			break;
		} else
		if (bUp)
		{
			menu_beep();
			if (activeLineIndexMisc == 0) 
				activeLineIndexMisc = lineCount;
			activeLineIndexMisc--;
			waitTime = 150;
		} else
		if (bDown)
		{
			menu_beep();
			activeLineIndexMisc++;
			if (activeLineIndexMisc == lineCount) 
				activeLineIndexMisc = 0;			
			waitTime = 150;
		}
	}
}

int activeLineIndexMain = 0;

void process_main_menu()
{
	const float lineWidth = 250.0;
	const int lineCount = 7;	

	std::string caption = "NATIVE  TRAINER  (AB)";

	static LPCSTR lineCaption[lineCount] = {
		"ADAM",
		"GUNS",
		"VEHICLE",
		"WORLD",
		"TIME",
		"WEATHER",
		"MISC"
	};

	DWORD waitTime = 150;
	while (true)
	{
		// timed menu draw, used for pause after active line switch
		DWORD maxTickCount = GetTickCount() + waitTime;
		do 
		{
			// draw menu
			draw_menu_line(caption, lineWidth, 15.0, 18.0, 0.0, 5.0, false, true);
			for (int i = 0; i < lineCount; i++)
				if (i != activeLineIndexMain)
					draw_menu_line(lineCaption[i], lineWidth, 9.0, 60.0 + i * 36.0, 0.0, 9.0, false, false);
			draw_menu_line(lineCaption[activeLineIndexMain], lineWidth + 1.0, 11.0, 56.0 + activeLineIndexMain * 36.0, 0.0, 7.0, true, false);
			
			update_features();
			WAIT(0);
		} while (GetTickCount() < maxTickCount);
		waitTime = 0;

		// process buttons
		bool bSelect, bBack, bUp, bDown;
		get_button_state(&bSelect, &bBack, &bUp, &bDown, NULL, NULL);
		if (bSelect)
		{
			menu_beep();
			switch (activeLineIndexMain)
			{
				case 0:
					process_player_menu();					
					break;
				case 1:
					process_weapon_menu();
					break;
				case 2:
					process_veh_menu();
					break;
				case 3:
					process_world_menu();
					break;
				case 4:
					process_time_menu();
					break;
				case 5:
					process_weather_menu();
					break;
				case 6:
					process_misc_menu();
					break;
			}
			waitTime = 200;
		} else
		if (bBack || trainer_switch_pressed())
		{
			menu_beep();
			break;
		} else
		if (bUp)
		{
			menu_beep();
			if (activeLineIndexMain == 0) 
				activeLineIndexMain = lineCount;
			activeLineIndexMain--;
			waitTime = 150;
		} else
		if (bDown)
		{
			menu_beep();
			activeLineIndexMain++;
			if (activeLineIndexMain == lineCount) 
				activeLineIndexMain = 0;			
			waitTime = 150;
		}
	}
}

void reset_globals()
{
	activeLineIndexMain			=
	activeLineIndexPlayer		=
	skinchangerActiveLineIndex	=
	skinchangerActiveItemIndex	=
	teleportActiveLineIndex		=
	activeLineIndexWeapon		=
	activeLineIndexVeh			=
	carspawnActiveLineIndex		=
	carspawnActiveItemIndex		=
	activeLineIndexWorld		=
	activeLineIndexWeather		=	0;

	featurePlayerInvincible				=
	featurePlayerInvincibleUpdated		=
	featurePlayerNeverWanted			=
	featurePlayerIgnored				=
	featurePlayerIgnoredUpdated			=
	featurePlayerUnlimitedAbility		=
	featurePlayerNoNoise				=
	featurePlayerNoNoiseUpdated			=
	featurePlayerFastSwim				=
	featurePlayerFastSwimUpdated		=
	featurePlayerFastRun				=
	featurePlayerFastRunUpdated			=
	featurePlayerSuperJump				=
	featureWeaponNoReload				=
	featureWeaponFireAmmo				=
	featureWeaponExplosiveAmmo			=
	featureWeaponExplosiveMelee			=
	featureWeaponVehRockets				=
	featureVehInvincible				=
	featureVehInvincibleUpdated			=
	featureVehInvincibleWheels			=
	featureVehInvincibleWheelsUpdated	=
	featureVehSeatbelt					=
	featureVehSeatbeltUpdated			=
	featureVehSpeedBoost				=
	featureVehWrapInSpawned				=
	featureWorldMoonGravity				=
	featureTimePaused					=
	featureTimePausedUpdated			=
	featureTimeSynced					=
	featureWeatherWind					=
	featureWeatherPers					=
	featureMiscLockRadio				=
	featureMiscHideHud					=	false;

	featureWorldRandomCops		=
	featureWorldRandomTrains	=
	featureWorldRandomBoats		=
	featureWorldGarbageTrucks	=	true;

	skinchanger_used			=	false;
}

int main()
{	
	reset_globals();

	while (true)
	{
		if (trainer_switch_pressed())
		{
			// Toggle Model
			
		}

		update_features();		
		WAIT(0);
	}

	return 0;
}

void ScriptMain()
{
	srand(GetTickCount());
	main();
}
