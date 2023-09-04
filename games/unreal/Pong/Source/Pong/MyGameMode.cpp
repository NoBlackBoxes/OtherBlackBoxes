#include "MyGameMode.h"
#include "MyPlayerController.h"
#include "MyPawn.h"

AMyGameMode::AMyGameMode()
{
	// Use our PlayerController class
	PlayerControllerClass = AMyPlayerController::StaticClass();

    // Use our character as the default pawn (or none?)
    DefaultPawnClass = AMyPawn::StaticClass();
}