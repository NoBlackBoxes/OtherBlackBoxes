// Copyright Epic Games, Inc. All Rights Reserved.

#include "QWOPGameMode.h"
#include "QWOPCharacter.h"
#include "UObject/ConstructorHelpers.h"

AQWOPGameMode::AQWOPGameMode()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnBPClass(TEXT("/Game/ThirdPerson/Blueprints/BP_ThirdPersonCharacter"));
	if (PlayerPawnBPClass.Class != NULL)
	{
		DefaultPawnClass = PlayerPawnBPClass.Class;
	}
}
