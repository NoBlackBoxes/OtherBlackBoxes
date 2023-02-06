#include "MyEnemy.h"
#include "MyActor.h"
#include "Components/StaticMeshComponent.h"
#include "Components/BoxComponent.h"

// Sets default values
AMyEnemy::AMyEnemy()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	// Create static mesh, set as root
	MyEnemyMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MyEnemyMesh"));
	SetRootComponent(MyEnemyMesh);

	// Create box collider, attach to root (mesh)
	MyEnemyCollider = CreateDefaultSubobject<UBoxComponent>(TEXT("MYEnemyCollider"));
	MyEnemyCollider->SetNotifyRigidBodyCollision(true);
	MyEnemyCollider->BodyInstance.SetCollisionProfileName("OverlapAll");
	MyEnemyCollider->SetupAttachment(GetRootComponent());
}

// Called when the game starts or when spawned
void AMyEnemy::BeginPlay()
{
	Super::BeginPlay();
	FVector Location = GetActorLocation();
	position = Location.Y;
	speed = 0.0f;
}

// Called every frame
void AMyEnemy::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	// Measure Y difference with ball
	FVector ActorLocation = MyActorToFollow->GetActorLocation();
	FVector EnemyLocation = GetActorLocation();
	speed = ActorLocation.Y - EnemyLocation.Y;
	position = EnemyLocation.Y;
	position += (10 * speed * DeltaTime);
	EnemyLocation.Y = position;

	// Clamp
	if(EnemyLocation.Y > 400)
	{
		EnemyLocation.Y = 400.0f;
	}
	if(EnemyLocation.Y < -400)
	{
		EnemyLocation.Y = -400.0f;
	}

	SetActorLocation(EnemyLocation);
}

