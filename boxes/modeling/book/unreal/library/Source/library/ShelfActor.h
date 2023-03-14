#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BookActor.h"
#include "ShelfActor.generated.h"

UCLASS()
class LIBRARY_API AShelfActor : public AActor
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, Category="Shelf Variables")
	class ABookActor *BookActor;
	
public:	
	// Sets default values for this actor's properties
	AShelfActor();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
