using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class basketball : MonoBehaviour
{
    public Camera main_cam;
    public Camera shot_cam;
    public AudioSource bounce_source;
    bool shooting = false;
    Vector2 rotation = Vector2.zero;

    // Start is called before the first frame update
    void Start()
    {
        // Set default camera
        main_cam.enabled = true;
        shot_cam.enabled = false;
        bounce_source = GetComponent<AudioSource>();
        
    }

    // Update is called once per frame
    void Update()
    {
        // Reset
        if(Input.GetKeyDown("r"))
        {    
            shooting = false;

            // Set default camera
            main_cam.enabled = true;
            shot_cam.enabled = false;
            
            transform.position = new Vector3(0f, 10f, 0f);
            GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
            GetComponent<Rigidbody>().useGravity = true;
        }

        // Dribble
        if(Input.GetKeyDown("d"))
        {
            float thrust = -1000.0f;
            GetComponent<Rigidbody>().AddRelativeForce(0f,thrust,0f,ForceMode.Acceleration);
        }

        // Shoot
        if(Input.GetKeyDown("s"))
        {   
            // Already shooting? Take the shot!
            if(shooting)
            {
                shooting = false;

                // Shoot!
                Vector3 shot_vector = new Vector3(0.0f, 0.0f, 1000.0f);                            
                GetComponent<Rigidbody>().AddRelativeForce(shot_vector);
                GetComponent<Rigidbody>().useGravity = true;
 
                // Return main camera
                main_cam.enabled = true;
                shot_cam.enabled = false;                
            }
            else
            {
                shooting = true;
            
                // Set shot camera
                main_cam.enabled = false;
                shot_cam.enabled = true;
                
                // Stop ball
                GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
                GetComponent<Rigidbody>().angularVelocity = new Vector3(0f, 0f, 0f);
                GetComponent<Rigidbody>().useGravity = false;
            }
        }
        
        // If shooting, mouse look aiming
        if(shooting)
        {
            rotation.y += Input.GetAxis ("Mouse X") * 3.0f;
            rotation.x += -Input.GetAxis ("Mouse Y") * 3.0f;
            transform.eulerAngles = (Vector2)rotation;
        }
    }
    
    // Add this function (for collision handling)
    void OnCollisionEnter (Collision collision)
    {
        // Get collision speed and set sound volume
        bounce_source.volume = Mathf.Clamp01(collision.relativeVelocity.magnitude / 20);

        // Play sound effect
        bounce_source.Play();
    }

}
