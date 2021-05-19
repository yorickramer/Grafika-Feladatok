//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kalmar Gabor Gyorgy
// Neptun : XCCGBS
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders


vec2 MiddleC(vec2 p1, vec2 p2) {

	float c1 = (p1 - p2).x;
	float c2 = (p1 - p2).y;
	float c3 = dot((p1 - p2), ((p1 + p2) / 2));
	float c4;
	float c5;
	float c6;

	if (length(p1) == 1.0f) {
		c4 = p1.x;
		c5 = p1.y;
		c6 = dot(p1, p1);
	}
	else if(length(p2) == 1.0f){
		c4 = p2.x;
		c5 = p2.y;
		c6 = dot(p2, p2);
		
	}
	else {
		vec2 p_star = p1 / (p1.x * p1.x + p1.y * p1.y);
		c4 = (p_star - p1).x;
		c5 = (p_star - p1).y;
		c6 = dot((p_star - p1), ((p_star + p1) / 2));
	}

	float div = c1 / c4;
	c1 = c1 - div * c4; 
	c2 = c2 - div * c5;
	c3 = c3 - div * c6;
	float y = c3 / c2;
	float x = (c6 - y * c5) / c4; 

	return vec2(x, y);
}


bool PolygonClockwise(const std::vector<vec2>& vertices){
	
	float area = 0.0f;
	for (int i = 0; i < vertices.size(); i++){
		int j = (i + 1) % vertices.size();
		area = area + 0.5 * (vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y);
	}
	if (area > 0.0f)
		return true;
	return false;
}


bool InsideTriangle(vec2 A, vec2 B, vec2 C, vec2 P) {

	std::vector<vec2> PAB{P, A, B};
	std::vector<vec2> PBC{P, B, C};
	std::vector<vec2> PCA{P, C, A};
	return ((PolygonClockwise(PAB) && PolygonClockwise(PBC) && PolygonClockwise(PCA)) || ((!PolygonClockwise(PAB) && !PolygonClockwise(PBC) && !PolygonClockwise(PCA))));
}


bool Snipable(const std::vector<vec2>& vertices, int Seq[], int a, int b, int c, int n){
	
	vec2 A = vertices[Seq[a]];
	vec2 B = vertices[Seq[b]];
	vec2 C = vertices[Seq[c]];

	if(0 > (((B.x - A.x) * (C.y - A.y)) - ((B.y - A.y) * (C.x - A.x))))
		return false;

	for (int i = 0; i < n; i++){
		if (((i != a) || (i != b) || (i != c)) && InsideTriangle(A, B, C, vertices[Seq[i]]))
			return false;
	}
	return true;
}


void Triangulate(const std::vector<vec2>& vertices, std::vector<vec2>& triangles) {
	int n = vertices.size();
	int Seq[300];

	if (PolygonClockwise(vertices))
		for (int i = 0; i < n; i++)
			Seq[i] = i;
	else
		for (int i = 0; i < n; i++) 
			Seq[i] = (n - 1) - i;

	int a_seq, c_seq;
	int b_seq = 0;
	while (n > 2) {

		a_seq = b_seq;
		if(a_seq == n)
			a_seq = 0;

		b_seq = a_seq + 1;
		if(b_seq == n)
			b_seq = 0;

		c_seq = b_seq + 1;
		if(c_seq == n)
			c_seq = 0;

		if (Snipable(vertices, Seq, a_seq, b_seq, c_seq, n))
		{
			triangles.push_back(vertices[Seq[a_seq]]);
			triangles.push_back(vertices[Seq[b_seq]]);
			triangles.push_back(vertices[Seq[c_seq]]);

			for (int i = b_seq; i < n-1; i++)
				Seq[i] = Seq[i+1];
			n = n-1;
		}
	}
}

void Perimeter(const vec2 *tomb) {
	
	float length = 0.0f;
	float lengthOfSide = 0.0f;
	float dx, dy, ds;

	for (int i = 0; i < 299; i++) {
		dx = tomb[i + 1].x - tomb[i].x;
		dy = tomb[i + 1].y - tomb[i].y;
		ds = sqrtf(dx * dx + dy * dy) / (1 - tomb[i].x * tomb[i].x - tomb[i].y * tomb[i].y);
		length = length + ds;
		if (i == 99) {
			printf("A - B oldal hossza: %f\n", length-lengthOfSide);
			lengthOfSide = length;
		}
		if (i == 199) {
			printf("B - C oldal hossza: %f\n", length - lengthOfSide);
			lengthOfSide = length;
		}
	}
	dx = tomb[0].x - tomb[299].x;
	dy = tomb[0].y - tomb[299].y;
	ds = sqrtf(dx * dx + dy * dy) / (1 - tomb[299].x * tomb[299].x - tomb[299].y * tomb[299].y);
	length = length + ds;
	printf("C - A oldal hossza: %f\n", length - lengthOfSide);
}

vec2 Angles(vec2 point1, vec2 point2, vec2 origo) {

	float fi_1 = atan2f((point1 - origo).y, (point1 - origo).x);
	float fi_2 = atan2f((point2 - origo).y, (point2 - origo).x);

	if (fi_1 < 0) {
		fi_1 = fi_1 + 2 * M_PI;
	}
	if (fi_2 < 0) {
		fi_2 = fi_2 + 2 * M_PI;
	}
	return vec2(fi_1, fi_2);
}

vec2 NormalAngles(vec2 point1, vec2 point2, vec2 origo) {

	float fi_1 = atan2f((point1 - origo).x, (-1) * (point1 - origo).y);
	float fi_2 = atan2f((point2 - origo).x * (-1), (point2 - origo).y);

	if (fi_1 < 0) {
		fi_1 = fi_1 + 2 * M_PI;
	}
	if (fi_2 < 0) {
		fi_2 = fi_2 + 2 * M_PI;
	}
	return vec2(fi_1, fi_2);
}

void AnglePrint(vec2 fi) {

	if (fi.x > fi.y) {
		if (fi.x - fi.y > M_PI)
			printf("csucs belso szoge: %f\n", (2 * M_PI - (fi.x - fi.y)) * 180.0f / M_PI);
		else
			printf("csucs belso szoge: %f\n",  (fi.x - fi.y) * 180.0f / M_PI);
	}
	else
		if (fi.y - fi.x > M_PI)
			printf("csucs belso szoge: %f\n", (2 * M_PI - (fi.y - fi.x)) * 180.0f / M_PI);
		else
			printf("csucs belso szoge: %f\n", (fi.y - fi.x) * 180.0f / M_PI);
}

void InsideAngles(vec2 p1, vec2 p2, vec2 p3, vec2 c1, vec2 c2, vec2 c3) {

	if (InsideTriangle(c1, c2, c3, (p1 + p2 + p3) / 2)) { 
		
		vec2 fi = NormalAngles(c1, c2, p1);
		printf("A ");
		AnglePrint(fi);

		fi = NormalAngles(c1, c3, p2);
		printf("B ");
		AnglePrint(fi);

		fi = NormalAngles(c2, c3, p3);
		printf("C ");
		AnglePrint(fi);
	}
	else {
		vec2 a = Angles(c1, c2, p1);
		float fi_p1 = abs(a.x - a.y);
		if (fi_p1 > M_PI)
			fi_p1 = 2 * M_PI - fi_p1;

		a = Angles(c1, c3, p2);
		float fi_p2 = abs(a.x - a.y);
		if (fi_p2 > M_PI)
			fi_p2 = 2 * M_PI - fi_p2;

		a = Angles(c2, c3, p3);
		float fi_p3 = abs(a.x - a.y);
		if (fi_p3 > M_PI)
			fi_p3 = 2 * M_PI - fi_p3;
		
		if (fi_p1 > fi_p2 && fi_p1> fi_p3) {
			
			vec2 fi = NormalAngles(c1, c2, p1);
			printf("A ");
			AnglePrint(fi);
			
			fi = Angles(c1, c3, p2);
			printf("B ");
			AnglePrint(fi);

			fi = Angles(c2, c3, p3);
			printf("C ");
			AnglePrint(fi);

		}
		else if (fi_p2 > fi_p1 && fi_p2 > fi_p3) {

			vec2 fi = Angles(c1, c2, p1);
			printf("A ");
			AnglePrint(fi);

			fi = NormalAngles(c1, c3, p2);
			printf("B ");
			AnglePrint(fi);

			fi = Angles(c2, c3, p3);
			printf("C ");
			AnglePrint(fi);
		}
		else {
			vec2 fi = Angles(c1, c2, p1);
			printf("A ");
			AnglePrint(fi);

			fi = Angles(c1, c3, p2);
			printf("B ");
			AnglePrint(fi);

			fi = NormalAngles(c2, c3, p3);
			printf("C ");
			AnglePrint(fi);
		}
	}
}


class Circle {
	unsigned int vao1;
	const int nv = 100;

public:
	void create() {
		glGenVertexArrays(1, &vao1);
		glBindVertexArray(vao1);
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		vec2 vertices[100];
		for (int i = 0; i < nv; i++) {
			float fi = i * 2 * M_PI / nv;
			vertices[i] = vec2(cosf(fi), sinf(fi));
		}
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * nv,  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {

		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 0.10f, 0.10f, 0.20f); // 3 floats

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);
		glBindVertexArray(vao1);
		glDrawArrays(GL_TRIANGLE_FAN, 0, nv); //startidx - elements
	}
};


class Triangle_Fill {
	unsigned int vao2;
	int count;
public:
	void create(std::vector<vec2>& vertices) {
		glGenVertexArrays(1, &vao2);
		glBindVertexArray(vao2);
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * vertices.size(),  // # bytes
			&vertices[0],	      	// address
			GL_STATIC_DRAW);	// we do not change later

		count = vertices.size();

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL); 

	}
	void Draw() {
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 0.20f, 0.60f, 1.00f); // 3 floats

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
									  0, 1, 0, 0,    // row-major!
									  0, 0, 1, 0,
									  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the 
		glBindVertexArray(vao2);
		glDrawArrays(GL_TRIANGLES, 0, count);
	}
};


class Triangle_Outline {
	unsigned int vao3;
	std::vector<vec2> vertices;
	
public:

	std::vector<vec2> GetVertices() { return vertices; }

	float Step(float fi_1, float fi_2) {
		
		float step;

		if (fi_2 > fi_1) {
			if ((fi_2 - fi_1) > M_PI) {
				step = ((-1) * (2 * M_PI - (fi_2 - fi_1))) / 100;
			}
			else {
				step = (fi_2 - fi_1) / 100;
			}
		}
		else {
			if ((fi_1 - fi_2) > M_PI) {
				step = (2 * M_PI - (fi_1 - fi_2)) / 100;
			}
			else {
				step = (-1) * (fi_1 - fi_2) / 100;
			}
		}
		return step;
	}

	void VerticesFill(vec2 c, float r, vec2 angles, std::vector<vec2> &vertices) {
		float fi_1 = angles.x;
		float fi_2 = angles.y;
		float t = fi_1;
		float step = Step(fi_1, fi_2);
		for (int i = 0; i < 100; i++) {
			vertices.push_back(vec2(c.x + r * cosf(t), c.y + r * sinf(t)));
			t = t + step;
		}
	}

	void create(vec2 p1, vec2 p2, vec2 p3) 
	{
		glGenVertexArrays(1, &vao3);
		glBindVertexArray(vao3);
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		vec2 c1 = MiddleC(p1, p2);
		vec2 c2 = MiddleC(p1, p3);
		vec2 c3 = MiddleC(p2, p3);

		float r_1 = sqrtf(dot(c1, c1) - 1);
		float r_2 = sqrtf(dot(c2, c2) - 1);
		float r_3 = sqrtf(dot(c3, c3) - 1);

		vertices.clear();

		vec2 angles = Angles(p1, p2, c1);
		VerticesFill(c1, r_1, angles, vertices);

		angles = Angles(p2, p3, c3);
		VerticesFill(c3, r_3, angles, vertices);
		
		angles = Angles(p1, p3, c2);
		VerticesFill(c2, r_2, vec2(angles.y,angles.x), vertices);
		
		printf("\n");
		Perimeter(&vertices[0]);
		printf("\n");
		InsideAngles(p1, p2, p3, c1, c2, c3);
		printf("----------------------------\n");
		
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * 300, &vertices[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {

		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 1.00f, 0.00f, 0.00f);

		float MVPtransf[4][4] = { 1, 0, 0, 0,
								  0, 1, 0, 0,
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);
		glBindVertexArray(vao3);
		glDrawArrays(GL_LINE_LOOP, 0, 300);
	}
};


Circle circle;
Triangle_Outline triangleOutline;
Triangle_Fill triangleFill;

class Container {
	vec2 tomb[3];
	int counter = 0;
	
public:
	void AddPoint(vec2 p) {
		tomb[counter] = p;
		if (counter == 2) {
			counter = -1;
			triangleOutline.create(tomb[0], tomb[1], tomb[2]);
			std::vector<vec2> triangles;
			Triangulate(triangleOutline.GetVertices(), triangles);
			triangleFill.create(triangles);
		}
		counter++;
	}
};

Container container;


void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(4.0f);
	circle.create();
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	circle.Draw();
	triangleOutline.Draw();
	triangleFill.Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}


void onMouse(int button, int state, int pX, int pY) { 

	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;

	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight;
		if(length(vec2(cX,cY)) <= 1)
			container.AddPoint(vec2(cX, cY));
		glutPostRedisplay();
	}
}

void onIdle() {
}
