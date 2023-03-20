#include <random>
#include "ShelfActor.h"
#include "BookActor.h"

// Sets default values
AShelfActor::AShelfActor()
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
}

// Called when the game starts or when spawned
void AShelfActor::BeginPlay()
{
	Super::BeginPlay();

	if (BookActorBP)
	{
		// Resize array
		BookActors.SetNum(num_books);

		// Spawn book actors
		FActorSpawnParameters SpawnInfo;
		for (size_t i = 0; i < num_books; i++)
		{
			FVector Location(0.f, 0.f, (float)(i*3.2)+3.14f);
//			FVector Location = RandomLocation();
			FRotator Rotation(0.0f, 0.0f, 0.0f);
			BookActors[i] = GetWorld()->SpawnActor<ABookActor>(BookActorBP, Location, Rotation);
		}
	}
}

// Called every frame
void AShelfActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

// Generate random position
FVector AShelfActor::RandomLocation()
{
	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<> pos_dist(-50, 50);
	std::uniform_real_distribution<> height_dist(0, 300);

	float x = pos_dist(e2);
	float y = pos_dist(e2);
	float z = height_dist(e2);
	FVector Location(x, y, z);

	return Location;
}