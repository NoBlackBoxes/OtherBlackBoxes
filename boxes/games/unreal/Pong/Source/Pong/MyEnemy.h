#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MyEnemy.generated.h"

UCLASS()
class PONG_API AMyEnemy : public AActor
{
	GENERATED_BODY()

public:	

	// Sets default values for this actor's properties
	AMyEnemy();
 
 	UPROPERTY(VisibleAnywhere, Category="My Variables")
	float position;

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	float speed;

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	class UStaticMeshComponent *MyEnemyMesh;

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	class UBoxComponent *MyEnemyCollider;

	UPROPERTY(EditAnywhere, Category="My Variables")
	class AMyActor *MyActorToFollow;

	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
