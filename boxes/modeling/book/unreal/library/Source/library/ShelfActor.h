#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BookActor.h"
#include "ShelfActor.generated.h"

UCLASS()
class LIBRARY_API AShelfActor : public AActor
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, Category = "Shelf Variables")
	int32 num_books;

	UPROPERTY(EditDefaultsOnly,Category="Shelf Variables")
	TSubclassOf<ABookActor> BookActorBP;

	UPROPERTY(Transient)
	TArray<ABookActor*> BookActors;

public:
	// Sets default values for this actor's properties
	AShelfActor();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;

private:
	FVector RandomLocation();
};
