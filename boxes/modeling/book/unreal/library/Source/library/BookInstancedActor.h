// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BookInstancedActor.generated.h"

UCLASS()
class LIBRARY_API ABookInstancedActor : public AActor
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, Category="Book Variables")
	class UInstancedStaticMeshComponent *BookInstancedMesh;

	UPROPERTY(EditAnywhere, Category="Book Variables")
	int32 num_rows;

	UPROPERTY(EditAnywhere, Category="Book Variables")
	int32 num_cols;

	UPROPERTY(Transient)
	TArray<FTransform> Transforms;

public:	
	// Sets default values for this actor's properties
	ABookInstancedActor();

	virtual void OnConstruction(const FTransform& Transform) override;

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
