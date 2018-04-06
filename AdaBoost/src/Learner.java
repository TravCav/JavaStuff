
public class Learner {
	
	public Learner()
		{
			this.Alpha = -1;
			this.Learned = false;
		}
	
		public double Alpha;

		public String Desc;

		public double Epsilon;

		public int Feature;

		public boolean Learned;

		public int Predicted;
		
		public String ToString()
		{
			return "Alpha: " + this.Alpha +
			       " | Description: " + this.Desc +
			       " | Epsilon: " + this.Epsilon +
			       " | Feature: " + this.Feature +
			       " | Learned: " + this.Learned +
			       " | Predicted: " + this.Predicted;
		}
		
		
		
}