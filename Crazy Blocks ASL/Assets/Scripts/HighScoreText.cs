using TMPro;
using UnityEngine;

public class HighScoreText : MonoBehaviour
{
    // Update is called once per frame
    void Update()
    {
        GetComponent<TMP_Text>().SetText($"{MovingBlock.highScore}");
    }
}
