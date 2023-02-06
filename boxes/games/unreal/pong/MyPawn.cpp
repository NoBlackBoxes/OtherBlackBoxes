#include "MyPawn.h"
#include "MyPlayerController.h"
#include "Components/StaticMeshComponent.h"
#include "Components/BoxComponent.h"
#include "EnhancedInputComponent.h"
#include "EnhancedInputSubsystems.h"

// Sets default values
AMyPawn::AMyPawn()
{
 	// Set this pawn to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	// Create static mesh, set as root
	MyPawnMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MyPawnMesh"));
	SetRootComponent(MyPawnMesh);

	// Create box collider, attach to root (mesh)
	MyPawnCollider = CreateDefaultSubobject<UBoxComponent>(TEXT("MyPawnCollider"));
	MyPawnCollider->SetNotifyRigidBodyCollision(true);
	MyPawnCollider->BodyInstance.SetCollisionProfileName("OverlapAll");
	MyPawnCollider->SetupAttachment(GetRootComponent());

}

// Called when the game starts or when spawned
void AMyPawn::BeginPlay()
{
	Super::BeginPlay();
	FVector Location = GetActorLocation();
	position = Location.Y;
	speed = 0.0f;
}

// Called every frame
void AMyPawn::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	FVector Location = GetActorLocation();
	position = Location.Y;
	position += (speed * DeltaTime);
	speed = speed / 1.55f;
	Location.Y = position;

	// Clamp
	if(Location.Y > 400)
	{
		Location.Y = 400.0f;
	}
	if(Location.Y < -400)
	{
		Location.Y = -400.0f;
	}

	SetActorLocation(Location);
}

// Called to bind functionality to input
void AMyPawn::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

	UEnhancedInputComponent* EIC = Cast<UEnhancedInputComponent>(PlayerInputComponent);
	AMyPlayerController* FPC = GetController<AMyPlayerController>();
	check(EIC && FPC);
	EIC->BindAction(FPC->MoveAction, ETriggerEvent::Triggered, this, &AMyPawn::Move);

	ULocalPlayer* LocalPlayer = FPC->GetLocalPlayer();
	check(LocalPlayer);

	UEnhancedInputLocalPlayerSubsystem* Subsystem = LocalPlayer->GetSubsystem<UEnhancedInputLocalPlayerSubsystem>();
	check(Subsystem);
	Subsystem->ClearAllMappings();
	Subsystem->AddMappingContext(FPC->PawnMappingContext, 0);
}

// Move
void AMyPawn::Move(const struct FInputActionValue& ActionValue)
{
	FVector Input = ActionValue.Get<FInputActionValue::Axis3D>();
	float direction = Input.X;
	UE_LOG(LogTemp, Warning, TEXT("MOVE: %.2f"), direction);
	speed = 500*Input.X;
}
