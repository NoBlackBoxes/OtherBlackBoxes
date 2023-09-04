#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "MyPawn.generated.h"

UCLASS()
class PONG_API AMyPawn : public APawn
{
	GENERATED_BODY()

public:

	// Sets default values for this pawn's properties
	AMyPawn();

	// Called to bind functionality to input
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	float position;

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	float speed;

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	class UStaticMeshComponent *MyPawnMesh;

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	class UBoxComponent *MyPawnCollider;

	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	// Called every frame
	virtual void Tick(float DeltaTime) override;

	/** Handle input to update location. */
	void Move(const struct FInputActionValue& ActionValue);

};
