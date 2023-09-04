#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BookActor.generated.h"

UCLASS()
class LIBRARY_API ABookActor final : public AActor
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, Category="Book Variables")
	class UStaticMeshComponent *BookMesh;

	UPROPERTY(EditAnywhere, Category="Book Variables")
	class UBoxComponent *BookCollider;
	
public:	
	// Sets default values for this actor's properties
	ABookActor();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
