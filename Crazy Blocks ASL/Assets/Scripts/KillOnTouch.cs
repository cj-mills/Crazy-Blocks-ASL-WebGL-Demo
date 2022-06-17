using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class KillOnTouch : MonoBehaviour
{
    void OnCollisionEnter2D(Collision2D collision)
    {
        if (collision.collider.CompareTag("Player"))
        {
            //string currentScene = SceneManager.GetActiveScene().name;
            //collision.collider.gameObject.GetComponent<Player>().Cleanup();
            collision.collider.gameObject.GetComponent<Player>().ResetPosition();
            MovingBlock[] movingBlocks = GameObject.Find("Moving Blocks").GetComponentsInChildren<MovingBlock>();
            foreach(MovingBlock movingBlock in movingBlocks)
            {
                movingBlock.ResetPosition();
            }

            //SceneManager.UnloadSceneAsync(currentScene);
            //SceneManager.LoadScene(currentScene);
        }
    }
}
