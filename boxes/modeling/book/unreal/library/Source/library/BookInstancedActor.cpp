#include "BookInstancedActor.h"
#include "Components/InstancedStaticMeshComponent.h"

// Sets default values
ABookInstancedActor::ABookInstancedActor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	BookInstancedMesh = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("BookInstancedMesh"));
	SetRootComponent(BookInstancedMesh);
}

// Called when the game starts or when spawned
void ABookInstancedActor::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ABookInstancedActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

// Called on construction
void ABookInstancedActor::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);

	float row_offset = 3.0f;
	float col_offset = 17.0f;

	if(BookInstancedMesh->GetInstanceCount() == 0)
	{
		Transforms.Empty(num_rows*num_cols);
		for (int r = 0; r < num_rows; r++)
		{
			for (int c = 0; c < num_cols; c++)
			{
				Transforms.Add(FTransform(FVector(col_offset * c, 0.f, row_offset * r)));
			}
		}

		BookInstancedMesh->AddInstances(Transforms, false);
	}
}