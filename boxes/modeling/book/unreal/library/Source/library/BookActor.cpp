#include "BookActor.h"
#include "Components/StaticMeshComponent.h"
#include "Components/BoxComponent.h"

// Sets default values
ABookActor::ABookActor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	BookMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("BookMesh"));
	SetRootComponent(BookMesh);

	BookCollider = CreateDefaultSubobject<UBoxComponent>(TEXT("BookCollider"));
	BookCollider->SetNotifyRigidBodyCollision(true);
	BookCollider->BodyInstance.SetCollisionProfileName("OverlapAll");
	BookCollider->SetupAttachment(GetRootComponent());
}

// Called when the game starts or when spawned
void ABookActor::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ABookActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

// Called on construction
void ABookActor::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);

	float row_offset = 3.0f;
	float col_offset = 17.0f;

	//if(BookMesh->GetInstanceCount() == 0)
	//{
	//	Transforms.Empty(num_rows*num_cols);
	//	for (int r = 0; r < num_rows; r++)
	//	{
	//		for (int c = 0; c < num_cols; c++)
	//		{
	//			Transforms.Add(FTransform(FVector(col_offset * c, 0.f, row_offset * r)));
	//		}
	//	}
	//
	//	BookMesh->AddInstances(Transforms, false);
	//}
}

