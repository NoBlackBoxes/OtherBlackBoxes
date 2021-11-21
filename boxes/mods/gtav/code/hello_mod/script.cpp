/*
	F6 key		advance time
*/

#include "script.h"
#include "keyboard.h"

#include <string>
#include <ctime>

int main()
{	
	// Locals
	int current_game_hour;
	int new_game_hour;

	// Loop
	while (true)
	{
		// Check for key (F6) press/release
		bool press = IsKeyJustUp(VK_F6);
		if (press)
		{
			// Get game time (in hours)
			current_game_hour = TIME::GET_CLOCK_HOURS();

			// Advance game time
			new_game_hour = (current_game_hour + 1) % 24;
			TIME::SET_CLOCK_TIME(new_game_hour, 0, 0);
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
