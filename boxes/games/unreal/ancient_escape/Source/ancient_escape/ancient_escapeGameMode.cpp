// Copyright Epic Games, Inc. All Rights Reserved.

#include "ancient_escapeGameMode.h"
#include "ancient_escapeCharacter.h"
#include "UObject/ConstructorHelpers.h"

Aancient_escapeGameMode::Aancient_escapeGameMode()
	: Super()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnClassFinder(TEXT("/Game/FirstPerson/Blueprints/BP_FirstPersonCharacter"));
	DefaultPawnClass = PlayerPawnClassFinder.Class;

}
