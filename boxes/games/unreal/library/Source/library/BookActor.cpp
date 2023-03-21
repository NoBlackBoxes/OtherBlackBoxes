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
