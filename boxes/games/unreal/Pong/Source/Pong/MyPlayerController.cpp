#include "MyPlayerController.h"
#include "InputAction.h"
#include "InputMappingContext.h"
#include "InputModifiers.h"

void AMyPlayerController::SetupInputComponent()
{
	// set up gameplay key bindings
	Super::SetupInputComponent();

	// Create these objects here and not in constructor because we only need them on the client.
	PawnMappingContext = NewObject<UInputMappingContext>(this);
	MoveAction = NewObject<UInputAction>(this);
	MoveAction->ValueType = EInputActionValueType::Axis3D;
	PawnMappingContext->MapKey(MoveAction, EKeys::D);

	FEnhancedActionKeyMapping &Mapping = PawnMappingContext->MapKey(MoveAction, EKeys::A);
	UInputModifierNegate *Negate = NewObject<UInputModifierNegate>(this);
	Mapping.Modifiers.Add(Negate);
}

void AMyPlayerController::OnInput()
{
}
