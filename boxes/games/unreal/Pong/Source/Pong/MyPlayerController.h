#pragma once

#include "CoreMinimal.h"
#include "GameFramework/PlayerController.h"
#include "MyPlayerController.generated.h"

UCLASS()
class AMyPlayerController : public APlayerController
{
	GENERATED_BODY()

public:

	virtual void SetupInputComponent() override;

	/** MappingContext */
	UPROPERTY()
	class UInputMappingContext* PawnMappingContext;
	
	/** Action to update location. */
	UPROPERTY()
	class UInputAction* MoveAction;

	void OnInput();

private:

};


