#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MyActor.generated.h"

UCLASS()
class PONG_API AMyActor : public AActor
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	FString bumper;

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	float ActorLifetime;

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	float X;

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	float Y;

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	float dX;

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	float dY;

	UPROPERTY(EditAnywhere, Category="My Variables")
	int32 ActorLevel;

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	class UStaticMeshComponent *MyActorMesh;

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	class USphereComponent *MyActorCollider;

	UPROPERTY(EditAnywhere, Category="My Variables")
	class USoundCue *MyActorSoundCue;

	UPROPERTY(VisibleAnywhere, Category="My Variables")
	class UAudioComponent *MyActorAudio;

public:	
	// Sets default values for this actor's properties
	AMyActor();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	UFUNCTION()
	void OnOverlap(
		UPrimitiveComponent* OverlappedComponent, 
    	AActor* OtherActor, 
    	UPrimitiveComponent* OtherComp, 
    	int32 OtherBodyIndex, 
    	bool bFromSweep, 
    	const FHitResult &SweepResult);
};
