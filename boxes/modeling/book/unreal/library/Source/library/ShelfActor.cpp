#include "ShelfActor.h"
#include "BookActor.h"

// Sets default values
AShelfActor::AShelfActor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	BookActor = CreateDefaultSubobject<ABookActor>(TEXT("BookActor"));

}

// Called when the game starts or when spawned
void AShelfActor::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AShelfActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

