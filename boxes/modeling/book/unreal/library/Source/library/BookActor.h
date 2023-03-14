#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BookActor.generated.h"

UCLASS()
class LIBRARY_API ABookActor final : public AActor
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, Category="Book Variables")
	class UStaticMeshComponent *BookMesh;

	UPROPERTY(VisibleAnywhere, Category="Book Variables")
	class UBoxComponent *BookCollider;

	UPROPERTY(EditAnywhere, Category="Book Variables")
	int32 num_rows;

	UPROPERTY(EditAnywhere, Category="Book Variables")
	int32 num_cols;

	UPROPERTY(Transient)
	TArray<FTransform> Transforms;
	
public:	
	// Sets default values for this actor's properties
	ABookActor();

	virtual void OnConstruction(const FTransform& Transform) override;

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
