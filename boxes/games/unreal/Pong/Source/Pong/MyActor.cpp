#include "MyActor.h"
#include "MyPawn.h"
#include "Components/StaticMeshComponent.h"
#include "Components/SphereComponent.h"
#include "Sound/SoundCue.h"
#include "Components/AudioComponent.h"

// Sets default values
AMyActor::AMyActor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	MyActorMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MyActorMeshComp"));
	SetRootComponent(MyActorMesh);

	MyActorCollider = CreateDefaultSubobject<USphereComponent>(TEXT("MyActorCollider"));
	MyActorCollider->SetNotifyRigidBodyCollision(true);
	MyActorCollider->BodyInstance.SetCollisionProfileName("OverlapAll");
	MyActorCollider->SetupAttachment(GetRootComponent());

	// Sound Cue
	MyActorSoundCue = CreateDefaultSubobject<USoundCue>(TEXT("MyActorSoundCue"));

	// Audio Comp
	MyActorAudio = CreateDefaultSubobject<UAudioComponent>(TEXT("MyActorAudio"));
	MyActorAudio->bAutoActivate = false;
	MyActorAudio->SetupAttachment(GetRootComponent());

}

// Called when the game starts or when spawned
void AMyActor::BeginPlay()
{
	Super::BeginPlay();
	UE_LOG(LogTemp, Warning, TEXT("Instance!"));

	// Set sound effect
	if (MyActorSoundCue->IsValidLowLevelFast()) {
		MyActorAudio->SetSound(MyActorSoundCue);
	}

	// Set initial speed
	dX = -400.0f;
	dY = 0.0f;

	// Set collision callback
	MyActorCollider->OnComponentBeginOverlap.AddDynamic(this, &AMyActor::OnOverlap);
}

// Called every frame
void AMyActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	ActorLifetime += DeltaTime;

	FVector Location = GetActorLocation();
	FRotator Rotation = GetActorRotation();
	//UE_LOG(LogTemp, Warning, TEXT("%s Location: (%f, %f, %f), Rotation: (%f, %f, %f)"), *GetName(), Location.X, Location.Y, Location.Z, Rotation.Pitch, Rotation.Yaw, Rotation.Roll );
	Location.X += dX * DeltaTime;
	Location.Y += dY * DeltaTime;
	X = Location.X;
	Y = Location.Y;

	// Score?
	if(abs(Location.X) > 550)
	{
		Location.X = 0.0f;
		Location.Y = 0.0f;
		dX = 400.0f;
		dY = 0.0f;
	}

	SetActorLocation(Location);
	SetActorRotation(Rotation);
}

// Called every frame
void AMyActor::OnOverlap(
	UPrimitiveComponent* OverlappedComponent, 
    AActor* OtherActor,
    UPrimitiveComponent* OtherComp, 
    int32 OtherBodyIndex, 
    bool bFromSweep, 
    const FHitResult &SweepResult)
{
	FString name = OtherActor->GetName();
	bumper = name;
	if(name.Mid(0,6).Equals("MyPawn"))
	{
		UE_LOG(LogTemp, Warning, TEXT("ME BUMP"));
		AMyPawn *other = (AMyPawn *) OtherActor;
		dY = other->speed * 1.5;
		
			// speed changing ball gets faster when collided 

		if(dX < 0)
		{
			dX = dX -40;
		}
		else
		{
			dX = dX + 40;
		}
		dX = -1 * dX;
	}
	else
	{
		if(OtherActor->IsRootComponentMovable())
		{
			UE_LOG(LogTemp, Warning, TEXT("THEM BUMP"));
			dX = -dX;
		}
		else
		{
			UE_LOG(LogTemp, Warning, TEXT("WALL BUMP"));
			dY = -dY;
		}
	}

	// Play sounds
	MyActorAudio->Play();
}